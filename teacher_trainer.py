from configuration import DatasetName, DatasetType, \
    AffectnetConf, D300wConf, W300Conf, InputDataSize, LearningConfig, CofwConf, WflwConf
from tf_record_utility import TFRecordUtility
from clr_callback import CyclicLR
from cnn_model import CNNModel
from custom_Losses import Custom_losses
from Data_custom_generator import CustomHeatmapGenerator
from PW_Data_custom_generator import PWCustomHeatmapGenerator
from image_utility import ImageUtility
import img_printer as imgpr

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from numpy import save, load, asarray
import csv
from skimage.io import imread
import pickle


# tf.compat.v1.enable_eager_execution()


class TeacherTrainer:

    def __init__(self, dataset_name, use_augmneted):
        self.dataset_name = dataset_name

        if dataset_name == DatasetName.w300:
            self.num_landmark = D300wConf.num_of_landmarks * 2
            if use_augmneted:
                self.img_path = D300wConf.augmented_train_image
                self.annotation_path = D300wConf.augmented_train_annotation
            else:
                self.img_path = D300wConf.no_aug_train_image
                self.annotation_path = D300wConf.no_aug_train_annotation

        if dataset_name == DatasetName.cofw:
            self.num_landmark = CofwConf.num_of_landmarks * 2
            self.img_path = CofwConf.augmented_train_image
            self.annotation_path = CofwConf.augmented_train_annotation

        if dataset_name == DatasetName.wflw:
            self.num_landmark = WflwConf.num_of_landmarks * 2
            if use_augmneted:
                self.img_path = WflwConf.augmented_train_image
                self.annotation_path = WflwConf.augmented_train_annotation
            else:
                self.img_path = WflwConf.no_aug_train_image
                self.annotation_path = WflwConf.no_aug_train_annotation

    def train(self, arch, weight_path):
        """"""
        '''create loss'''
        c_loss = Custom_losses()

        '''create summary writer'''
        summary_writer = tf.summary.create_file_writer(
            "./train_logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

        '''making models'''
        model = self.make_model(arch=arch, w_path=weight_path, is_old=False)

        '''create optimizer'''
        _lr = 1e-3
        optimizer_student = self._get_optimizer(lr=_lr)

        '''create sample generator'''
        x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames = self._create_generators()

        '''create train configuration'''
        step_per_epoch = len(x_train_filenames) // LearningConfig.batch_size

        '''start train:'''
        for epoch in range(LearningConfig.epochs):
            x_train_filenames, y_train_filenames = self._shuffle_data(x_train_filenames, y_train_filenames)
            for batch_index in range(step_per_epoch):
                '''load annotation and images'''
                images, annotation_gr = self._get_batch_sample(
                    batch_index=batch_index, x_train_filenames=x_train_filenames,
                    y_train_filenames=y_train_filenames, model=model)
                '''convert to tensor'''
                images = tf.cast(images, tf.float32)
                annotation_gr = tf.cast(annotation_gr, tf.float32)

                '''train step'''
                self.train_step(epoch=epoch, step=batch_index, total_steps=step_per_epoch, images=images,
                                model=model,
                                annotation_gr=annotation_gr,
                                optimizer=optimizer_student,
                                summary_writer=summary_writer, c_loss=c_loss)
            '''evaluating part'''
            img_batch_eval, pn_batch_eval = self._create_evaluation_batch(x_val_filenames, y_val_filenames)
            # loss_eval, loss_eval_tol_dif_stu, loss_eval_tol_dif_gt, loss_eval_tou_dif_stu, loss_eval_tou_dif_gt = \
            loss_eval = self._eval_model(img_batch_eval, pn_batch_eval, model)
            with summary_writer.as_default():
                tf.summary.scalar('Eval-LOSS', loss_eval, step=epoch)
            '''save weights'''
            model.save(
                './models/teacher_model_' + str(epoch) + '_' + self.dataset_name + '_' + str(loss_eval) + '.h5')

    # @tf.function
    def train_step(self, epoch, step, total_steps, images,
                   model, annotation_gr,
                   optimizer, summary_writer, c_loss):
        with tf.GradientTape() as tape:
            '''create annotation_predicted'''
            annotation_predicted = model(images, training=True)
            '''calculate loss'''
            loss = c_loss.MSE(x_pr=annotation_predicted, x_gt=annotation_gr)
            '''calculate gradient'''
            gradients = tape.gradient(loss, model.trainable_variables)
            '''apply Gradients:'''
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            '''printing loss Values: '''
            tf.print("->EPOCH: ", str(epoch), "->STEP: ", str(step) + '/' + str(total_steps), ' -> : LOSS: ', loss)
            with summary_writer.as_default():
                tf.summary.scalar('LOSS', loss, step=epoch)

    def make_model(self, arch, w_path, is_old=False):
        cnn = CNNModel()
        model = cnn.get_model(arch=arch, output_len=self.num_landmark, input_tensor=None, weight_path=w_path,
                              is_old=is_old)
        if w_path is not None and arch != 'mobileNetV2_d' and not is_old:
            model.load_weights(w_path)
        return model

    def _eval_model(self, img_batch_eval, pn_batch_eval, model):
        # annotation_predicted, pr_tol_dif_stu, pr_tol_dif_gt, pr_tou_dif_stu, pr_tou_dif_gt = model(img_batch_eval)
        annotation_predicted = model(img_batch_eval)
        loss_eval = np.array(tf.reduce_mean(tf.abs(pn_batch_eval - annotation_predicted)))
        # loss_eval_tol_dif_stu = np.array(tf.reduce_mean(tf.abs(pn_batch_eval - annotation_predicted)))
        # loss_eval_tol_dif_gt = np.array(tf.reduce_mean(tf.abs(pn_batch_eval - annotation_predicted)))
        # loss_eval_tou_dif_stu = np.array(tf.reduce_mean(tf.abs(pn_batch_eval - annotation_predicted)))
        # loss_eval_tou_dif_gt = np.array(tf.reduce_mean(tf.abs(pn_batch_eval - annotation_predicted)))
        # return loss_eval, loss_eval_tol_dif_stu, loss_eval_tol_dif_gt, loss_eval_tou_dif_stu, loss_eval_tou_dif_gt
        return loss_eval

    def _get_optimizer(self, lr=1e-2, beta_1=0.9, beta_2=0.999, decay=1e-4):
        return tf.keras.optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)

    def _shuffle_data(self, filenames, labels):
        filenames_shuffled, y_labels_shuffled = shuffle(filenames, labels)
        return filenames_shuffled, y_labels_shuffled

    def _create_generators(self):
        fn_prefix = './file_names/' + self.dataset_name + '_'
        # x_trains_path = fn_prefix + 'x_train_fns.npy'
        # x_validations_path = fn_prefix + 'x_val_fns.npy'

        tf_utils = TFRecordUtility(number_of_landmark=self.num_landmark)

        filenames, labels = tf_utils.create_image_and_labels_name(img_path=self.img_path,
                                                                  annotation_path=self.annotation_path)
        filenames_shuffled, y_labels_shuffled = shuffle(filenames, labels)
        x_train_filenames, x_val_filenames, y_train, y_val = train_test_split(
            filenames_shuffled, y_labels_shuffled, test_size=LearningConfig.batch_size, random_state=1)

        # save(x_trains_path, filenames_shuffled)
        # save(x_validations_path, y_labels_shuffled)

        # save(x_trains_path, x_train_filenames)
        # save(x_validations_path, x_val_filenames)
        # save(y_trains_path, y_train)
        # save(y_validations_path, y_val)

        # return filenames_shuffled, y_labels_shuffled
        return x_train_filenames, x_val_filenames, y_train, y_val

    def _create_evaluation_batch(self, x_eval_filenames, y_eval_filenames):
        img_path = self.img_path
        pn_tr_path = self.annotation_path
        '''create batch data and normalize images'''
        batch_x = x_eval_filenames[0:LearningConfig.batch_size]
        batch_y = y_eval_filenames[0:LearningConfig.batch_size]
        '''create img and annotations'''
        img_batch = np.array([imread(img_path + file_name) for file_name in batch_x]) / 255.0
        if self.dataset_name == DatasetName.cofw:  # this ds is not normalized
            pn_batch = np.array([load(pn_tr_path + file_name) for file_name in batch_y])
        else:
            pn_batch = np.array([self._load_and_normalize(pn_tr_path + file_name) for file_name in batch_y])
        return img_batch, pn_batch

    def _get_batch_sample(self, batch_index, x_train_filenames, y_train_filenames):
        img_path = self.img_path
        pn_tr_path = self.annotation_path
        '''create batch data and normalize images'''
        batch_x = x_train_filenames[
                  batch_index * LearningConfig.batch_size:(batch_index + 1) * LearningConfig.batch_size]
        batch_y = y_train_filenames[
                  batch_index * LearningConfig.batch_size:(batch_index + 1) * LearningConfig.batch_size]
        img_batch = np.array([imread(img_path + file_name) for file_name in batch_x]) / 255.0
        pn_batch = np.array([self._load_and_normalize(pn_tr_path + file_name) for file_name in batch_y])

        return img_batch, pn_batch

    def _load_and_normalize(self, point_path):
        annotation = load(point_path)

        """for training we dont normalize COFW"""

        '''normalize landmarks based on hyperface method'''
        width = InputDataSize.image_input_size
        height = InputDataSize.image_input_size
        x_center = width / 2
        y_center = height / 2
        annotation_norm = []
        for p in range(0, len(annotation), 2):
            annotation_norm.append((x_center - annotation[p]) / width)
            annotation_norm.append((y_center - annotation[p + 1]) / height)
        return annotation_norm
