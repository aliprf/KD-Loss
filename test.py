from configuration import DatasetName, DatasetType, \
    AffectnetConf, D300wConf, W300Conf, InputDataSize, LearningConfig, CofwConf, WflwConf
from tf_record_utility import TFRecordUtility
from image_utility import ImageUtility
from skimage.transform import resize
import numpy as np
import math

import cv2
import os.path
import scipy.io as sio
from cnn_model import CNNModel
import img_printer as imgpr
from tqdm import tqdm


class Test:
    def __init__(self, dataset_name, arch, num_output_layers, weight_fname, has_pose=False):
        self.dataset_name = dataset_name
        self.has_pose = has_pose

        if dataset_name == DatasetName.w300:
            self.output_len = D300wConf.num_of_landmarks * 2
        elif dataset_name == DatasetName.cofw:
            self.output_len = CofwConf.num_of_landmarks * 2
        elif dataset_name == DatasetName.wflw:
            self.output_len = WflwConf.num_of_landmarks * 2

        cnn = CNNModel()
        model = cnn.get_model(arch=arch, input_tensor=None, output_len=self.output_len)

        model.load_weights(weight_fname)

        img = None # load a cropped image

        image_utility = ImageUtility()
        pose_predicted = []
        image = np.expand_dims(img, axis=0)

        pose_predicted = model.predict(image)[1][0]