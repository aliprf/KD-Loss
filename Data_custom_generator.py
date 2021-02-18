from skimage.io import imread
from skimage.transform import resize
import numpy as np
import tensorflow as tf
import keras
from skimage.transform import resize
from tf_record_utility import TFRecordUtility
from configuration import DatasetName, DatasetType, \
    AffectnetConf, D300wConf, W300Conf, InputDataSize, LearningConfig
from numpy import save, load, asarray


class CustomHeatmapGenerator(keras.utils.Sequence):

    def __init__(self, is_single, image_filenames, label_filenames, batch_size, n_outputs, accuracy=100):
        self.image_filenames = image_filenames
        self.label_filenames = label_filenames
        self.batch_size = batch_size
        self.n_outputs = n_outputs
        self.is_single = is_single
        self.accuracy = accuracy

    def __len__(self):
        _len = np.ceil(len(self.image_filenames) // float(self.batch_size))
        return int(_len)

    def __getitem__(self, idx):
        img_path = D300wConf.train_images_dir
        tr_path_85 = D300wConf.train_hm_dir_85
        tr_path_90 = D300wConf.train_hm_dir_90
        tr_path_97 = D300wConf.train_hm_dir_97
        tr_path = D300wConf.train_hm_dir

        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.label_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        img_batch = np.array([imread(img_path + file_name) for file_name in batch_x])

        if self.is_single:
            if self.accuracy == 85:
                lbl_batch = np.array([load(tr_path_85 + file_name) for file_name in batch_y])
            elif self.accuracy == 90:
                lbl_batch = np.array([load(tr_path_90 + file_name) for file_name in batch_y])
            elif self.accuracy == 97:
                lbl_batch = np.array([load(tr_path_97 + file_name) for file_name in batch_y])
            else:
                lbl_batch = np.array([load(tr_path + file_name) for file_name in batch_y])

            lbl_out_array = lbl_batch
        else:
            lbl_batch_85 = np.array([load(tr_path_85 + file_name) for file_name in batch_y])
            lbl_batch_90 = np.array([load(tr_path_90 + file_name) for file_name in batch_y])
            lbl_batch_97 = np.array([load(tr_path_97 + file_name) for file_name in batch_y])
            lbl_out_array = [lbl_batch_85, lbl_batch_90, lbl_batch_97]

        return img_batch, lbl_out_array
