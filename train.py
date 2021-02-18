from configuration import DatasetName, DatasetType, D300wConf, InputDataSize, LearningConfig, CofwConf, WflwConf
from cnn_model import CNNModel
from custom_Losses import Custom_losses
import tensorflow as tf
from tensorflow import keras

print(tf.__version__)
print(keras.__version__)

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow.keras.callbacks import CSVLogger
from datetime import datetime
from sklearn.utils import shuffle
import os
from sklearn.model_selection import train_test_split
from numpy import save, load, asarray
import os.path
from tensorflow.keras import losses
from keras import backend as K
import csv
from skimage.io import imread


class Train:
    def __init__(self, dataset_name, arch, weight=None, accuracy=100):
        self.dataset_name = dataset_name

        if dataset_name == DatasetName.ibug:
            self.SUM_OF_ALL_TRAIN_SAMPLES = D300wConf.orig_number_of_training
            self.output_len = D300wConf.num_of_landmarks * 2
            if accuracy == 100:
                self.tf_train_path = D300wConf.augmented_train_tf_path + 'train100.tfrecords'
                self.tf_eval_path = D300wConf.augmented_train_tf_path + 'eval100.tfrecords'
            elif accuracy == 90:
                self.tf_train_path = D300wConf.augmented_train_tf_path + 'train90.tfrecords'
                self.tf_eval_path = D300wConf.augmented_train_tf_path + 'eval90.tfrecords'

        elif dataset_name == DatasetName.cofw:
            '''we use AUGmented data for teacher'''
            self.SUM_OF_ALL_TRAIN_SAMPLES = CofwConf.orig_number_of_training
            self.output_len = CofwConf.num_of_landmarks * 2
            if accuracy == 100:
                self.tf_train_path = CofwConf.augmented_train_tf_path + 'train100.tfrecords'
                self.tf_eval_path = CofwConf.augmented_train_tf_path + 'eval100.tfrecords'
            elif accuracy == 90:
                self.tf_train_path = CofwConf.augmented_train_tf_path + 'train90.tfrecords'
                self.tf_eval_path = CofwConf.augmented_train_tf_path + 'eval90.tfrecords'

        elif dataset_name == DatasetName.wflw:
            '''we use original data for teacher'''
            self.SUM_OF_ALL_TRAIN_SAMPLES = WflwConf.orig_number_of_training
            self.output_len = WflwConf.num_of_landmarks * 2
            if accuracy == 100:
                self.tf_train_path = WflwConf.augmented_train_tf_path + 'train100.tfrecords'
                self.tf_eval_path = WflwConf.augmented_train_tf_path + 'eval100.tfrecords'
            elif accuracy == 90:
                self.tf_train_path = WflwConf.no_aug_train_tf_path + 'train90.tfrecords'
                self.tf_eval_path = WflwConf.no_aug_train_tf_path + 'eval90.tfrecords'

        self.BATCH_SIZE = LearningConfig.batch_size
        self.STEPS_PER_VALIDATION_EPOCH = 10
        self.STEPS_PER_EPOCH = self.SUM_OF_ALL_TRAIN_SAMPLES // self.BATCH_SIZE
        self.EPOCHS = LearningConfig.epochs
        self.loss = losses.mean_squared_error
        self.arch = arch
        self.weight = weight
        self.accuracy = accuracy
        self.train_fit()

    def train_fit(self):
        '''prepare callbacks'''
        callbacks_list = self._prepare_callback()

        ''' define optimizers'''
        optimizer = self._get_optimizer()

        '''create train, validation, test data iterator'''
        train_images, train_landmarks = self.create_training_tensor_points(tfrecord_filename=self.tf_train_path,
                                                                           batch_size=self.BATCH_SIZE)
        validation_images, validation_landmarks = self.create_training_tensor_points(
            tfrecord_filename=self.tf_eval_path,
            batch_size=self.BATCH_SIZE)

        '''creating model'''
        cnn = CNNModel()

        model = cnn.get_model(arch=self.arch,
                              output_len=self.output_len,
                              input_tensor=train_images, inp_shape=None)
        if self.weight is not None:
            model.load_weights(self.weight)

        '''compiling model'''
        model.compile(loss=self._generate_loss(),
                      optimizer=optimizer,
                      metrics=['mse', 'mae'],
                      target_tensors=self._generate_target_tensors(train_landmarks),
                      loss_weights=self._generate_loss_weights()
                      )

        '''train Model '''
        print('< ========== Start Training ============= >')

        history = model.fit(train_images,
                            train_landmarks,
                            epochs=self.EPOCHS,
                            steps_per_epoch=self.STEPS_PER_EPOCH,
                            validation_data=(validation_images, validation_landmarks),
                            validation_steps=self.STEPS_PER_VALIDATION_EPOCH,
                            verbose=1, callbacks=callbacks_list
                            )

    def create_training_tensor_points(self, tfrecord_filename, batch_size):
        SHUFFLE_BUFFER = 100
        BATCH_SIZE = batch_size
        dataset = tf.data.TFRecordDataset(tfrecord_filename)
        # Maps the parser on every file path in the array. You can set the number of parallel loaders here
        dataset = dataset.map(self._parse_function_points, num_parallel_calls=32)
        # This dataset will go on forever
        dataset = dataset.repeat()
        # Set the number of data points you want to load and shuffle
        dataset = dataset.shuffle(SHUFFLE_BUFFER)
        # Set the batch size
        dataset = dataset.batch(BATCH_SIZE)
        # Create an iterator
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        # Create your tf representation of the iterator
        images, landmarks = iterator.get_next()
        return images, landmarks

    def _parse_function_points(self, proto):
        keys_to_features = {'landmarks': tf.compat.v1.FixedLenFeature([self.num_landmarks], tf.float32),
                            'image_raw': tf.compat.v1.FixedLenFeature([InputDataSize.image_input_size,
                                                                       InputDataSize.image_input_size, 3], tf.float32)}

        parsed_features = tf.compat.v1.parse_single_example(proto, keys_to_features)
        _images = parsed_features['image_raw']
        _landmarks = parsed_features["landmarks"]

        return _images, _landmarks

    def _generate_loss(self):
        return self.loss

    def _generate_target_tensors(self, target_tensor):
        return target_tensor

    def _generate_loss_weights(self):
        wights = [1]
        return wights

    def _get_optimizer(self):
        return Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, decay=1e-5, amsgrad=False)

    def _prepare_callback(self):
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=1000, verbose=1,
                                                   mode='min')
        file_path = "ds_" + str(self.dataset_name) + "_ac_" + str(self.accuracy) + "_weights-{epoch:02d}-{loss:.5f}.h5"
        checkpoint = ModelCheckpoint(file_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
        csv_logger = CSVLogger('log.csv', append=True, separator=';')
        # tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
        # return [checkpoint, early_stop, csv_logger, clr, tensorboard_callback]
        return [checkpoint, early_stop, csv_logger]

    def _save_model_with_weight(self, model):
        model_json = model.to_json()

        with open("model_asm_shrink.json", "w") as json_file:
            json_file.write(model_json)

        model.save_weights("model_asm_shrink.h5")
        print("Saved model to disk")

    def write_loss_log(self, file_name, row_data):
        with open(file_name, 'w') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(row_data)
