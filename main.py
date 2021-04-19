# The Iris ML Flask App

import pickle
from flask import Flask, render_template, request
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from PIL.Image import Image as PilImage
from utils.utils import *

from IPython.display import display
from IPython.display import Image as _Imgdis
import random
from clodsa.augmentors.augmentorFactory import createAugmentor
from clodsa.transformers.transformerFactory import transformerGenerator
from clodsa.techniques.techniqueFactory import createTechnique
import json
import matplotlib.patches as patches
import math
from skimage import io
import skimage.feature
import cv2
from tqdm import tqdm

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data.catalog import DatasetCatalog


app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/index.html', methods=['GET'])
def main_index():
    return render_template('index.html')

@app.route('/single-post.html', methods=['GET'])
def single_post():
    return render_template('single-post.html')

@app.route('/data_exploration.html', methods=['GET'])
def data_exploration():
    return render_template('data_exploration.html')

@app.route('/new_test.html', methods=['GET','POST'])
def new_test():
    upload_dir = os.path.join("..","img")
    if request.method == 'POST':
        # save the single "profile" file
        profile = request.files['profile']
        profile.save(os.path.join(uploads_dir, "test_image.png"))

        return redirect(url_for('upload'))
    return render_template('new_test.html')

@app.route('/prediction.html', methods=['GET'])
def prediction():
    files = [os.path.join("augmented_data", f) for f in os.listdir("augmented_data") if (os.path.isfile(os.path.join("augmented_data", f)) and  "csv" not in f and "json" not in f)]
    average=[]
    for file in files:
        img = io.imread(file)
        average.append(img.mean(axis=0).mean(axis=0))

    mean_pixel_values=np.array(average).mean(axis=0)
    std_pixel_values=np.array(average).std(axis=0)
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    #cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
    cfg.INPUT.MAX_SIZE_TRAIN = 2000
    cfg.INPUT.MIN_SIZE_TRAIN = 800
    cfg.INPUT.MAX_SIZE_TEST = 5000
    cfg.INPUT.MIN_SIZE_TEST = 1000
    cfg.MODEL.PIXEL_MEAN = [mean_pixel_values[2], mean_pixel_values[1], mean_pixel_values[0]]#
    cfg.MODEL.DEVICE="cuda"
    cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"  # initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.01
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8,16,32]]
    cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [[0,90]]
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.50   # set the testing threshold for this model
    img = cv2.imread("static/img/test_image.png")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(img)
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.5)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    fig,ax = plt.subplots(1,figsize=(20,20))
    plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    fig.savefig("static/img/result_image.png",orientation="landscape", bbox_inches="tight",pad_inches=0)


    return render_template('prediction.html')



if __name__ == '__main__':
    app.debug = True
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True)
