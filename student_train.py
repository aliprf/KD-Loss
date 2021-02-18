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


class StudentTrainer:

    def __init__(self, dataset_name, use_augmneted):
        self.dataset_name = dataset_name

        if dataset_name == DatasetName.ibug or dataset_name == DatasetName.w300:
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

    def train(self, arch_student, weight_path_student, loss_weight_student,
              arch_tough_teacher, weight_path_tough_teacher, loss_weight_tough_teacher,
              arch_tol_teacher, weight_path_tol_teacher, loss_weight_tol_teacher):
        """"""
        '''create loss'''
        c_loss = Custom_losses()

        '''create summary writer'''
        summary_writer = tf.summary.create_file_writer(
            "./train_logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

        '''making models'''
        model_student = self.make_model(arch=arch_student, w_path=weight_path_student, is_old=False)
        model_tough_teacher = self.make_model(arch=arch_tough_teacher, w_path=weight_path_tough_teacher)
        model_tol_teacher = self.make_model(arch=arch_tol_teacher, w_path=weight_path_tol_teacher)

        '''create optimizer'''
        _lr = 1e-3
        optimizer_student = self._get_optimizer(lr=_lr)

        '''create sample generator'''
        x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames = self._create_generators()
        # x_train_filenames, y_train_filenames = self._create_generators()

        '''create train configuration'''
        step_per_epoch = len(x_train_filenames) // LearningConfig.batch_size

        '''start train:'''
        for epoch in range(LearningConfig.epochs):
            x_train_filenames, y_train_filenames = self._shuffle_data(x_train_filenames, y_train_filenames)
            for batch_index in range(step_per_epoch):
                '''load annotation and images'''
                images, annotation_gr, annotation_tough_teacher, annotation_tol_teacher, annotation_student = self._get_batch_sample(
                    batch_index=batch_index, x_train_filenames=x_train_filenames,
                    y_train_filenames=y_train_filenames, model_tough_t=model_tough_teacher,
                    model_tol_t=model_tol_teacher, model_student=model_student)
                '''convert to tensor'''
                images = tf.cast(images, tf.float32)
                annotation_gr = tf.cast(annotation_gr, tf.float32)
                annotation_tough_teacher = tf.cast(annotation_tough_teacher, tf.float32)
                annotation_tol_teacher = tf.cast(annotation_tol_teacher, tf.float32)

                '''train step'''
                self.train_step(epoch=epoch, step=batch_index, total_steps=step_per_epoch, images=images,
                                model_student=model_student,
                                annotation_gr=annotation_gr, annotation_tough_teacher=annotation_tough_teacher,
                                annotation_tol_teacher=annotation_tol_teacher,
                                l_w_stu=loss_weight_student, l_w_togh_t=loss_weight_tough_teacher,
                                loss_w_tol_t=loss_weight_tol_teacher,
                                optimizer=optimizer_student, summary_writer=summary_writer, c_loss=c_loss)
            '''evaluating part'''
            img_batch_eval, pn_batch_eval = self._create_evaluation_batch(x_val_filenames, y_val_filenames)
            # loss_eval, loss_eval_tol_dif_stu, loss_eval_tol_dif_gt, loss_eval_tou_dif_stu, loss_eval_tou_dif_gt = \
            loss_eval = self._eval_model(img_batch_eval, pn_batch_eval, model_student)
            with summary_writer.as_default():
                tf.summary.scalar('Eval-LOSS', loss_eval, step=epoch)
                # tf.summary.scalar('Eval-loss_eval_tol_dif_stu', loss_eval_tol_dif_stu, step=epoch)
                # tf.summary.scalar('Eval-loss_eval_tol_dif_gt', loss_eval_tol_dif_gt, step=epoch)
                # tf.summary.scalar('Eval-loss_eval_tou_dif_stu', loss_eval_tou_dif_stu, step=epoch)
                # tf.summary.scalar('Eval-loss_eval_tou_dif_gt', loss_eval_tou_dif_gt, step=epoch)
            # model_student.save_weights('./models/stu_weight_' + '_' + str(epoch) + self.dataset_name + '_' + str(loss_eval) + '.h5')
            '''save weights'''
            model_student.save(
                './models/stu_model_' + str(epoch) + '_' + self.dataset_name + '_' + str(loss_eval) + '.h5')
            '''calculate Learning rate'''
            # _lr = self._calc_learning_rate(iterations=epoch, step_size=20, base_lr=1e-5, max_lr=1e-1)
            # optimizer_student = self._get_optimizer(lr=_lr)

    def _calc_learning_rate(self, iterations, step_size, base_lr, max_lr):
        cycle = np.floor(1 + iterations / (2 * step_size))
        x = np.abs(iterations / step_size - 2 * cycle + 1)
        lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x)) / float(2 ** (cycle - 1))
        print('LR is: ' + str(lr))
        return lr

    # @tf.function
    def train_step(self, epoch, step, total_steps, images, model_student, annotation_gr,
                   annotation_tough_teacher, annotation_tol_teacher, annotation_student,
                   l_w_stu, l_w_togh_t, loss_w_tol_t,
                   optimizer, summary_writer, c_loss, train_dif):
        with tf.GradientTape() as tape_student:
            '''create annotation_predicted'''
            # annotation_predicted, pr_tol, pr_tol_dif_gt, pr_tou, pr_tou_dif_gt = model_student(
            annotation_predicted = model_student(
                images, training=True)
            '''calculate loss'''
            loss_total, loss_main, loss_tough_assist, loss_tol_assist = c_loss.kd_loss(x_pr=annotation_predicted,
                                                                                       x_gt=annotation_gr,
                                                                                       x_tough=annotation_tough_teacher,
                                                                                       x_tol=annotation_tol_teacher,
                                                                                       alpha_tough=1.9,
                                                                                       alpha_mi_tough=-0.45,
                                                                                       alpha_tol=1.8,
                                                                                       alpha_mi_tol=-0.4,
                                                                                       main_loss_weight=l_w_stu,
                                                                                       tough_loss_weight=l_w_togh_t,
                                                                                       tol_loss_weight=loss_w_tol_t)
            '''calculate gradient'''
            gradients_of_student = tape_student.gradient(loss_total, model_student.trainable_variables)
            '''apply Gradients:'''
            optimizer.apply_gradients(zip(gradients_of_student, model_student.trainable_variables))
            '''printing loss Values: '''
            tf.print("->EPOCH: ", str(epoch), "->STEP: ", str(step) + '/' + str(total_steps),
                     ' -> : LOSS: ', loss_total,
                     ' -> : loss_main: ', loss_main,
                     ' -> : loss_tough_assist: ', loss_tough_assist,
                     ' -> : loss_tol_assist: ', loss_tol_assist)
            with summary_writer.as_default():
                tf.summary.scalar('LOSS', loss_total, step=epoch)
                tf.summary.scalar('loss_main', loss_main, step=epoch)
                tf.summary.scalar('loss_tough_assist', loss_tough_assist, step=epoch)
                tf.summary.scalar('loss_tol_assist', loss_tol_assist, step=epoch)

    def make_model(self, arch, w_path, is_old=False):
        cnn = CNNModel()
        model = cnn.get_model(arch=arch, output_len=self.num_landmark, input_tensor=None, weight_path=w_path,
                              is_old=is_old)
        if w_path is not None and arch != 'mobileNetV2_d' and not is_old:
            model.load_weights(w_path)
        # model.save('test_model'+arch+'.h5')
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

    def _get_batch_sample(self, batch_index, x_train_filenames, y_train_filenames, model_tough_t, model_tol_t, model_student,
                          train_dif):
        img_path = self.img_path
        pn_tr_path = self.annotation_path
        '''create batch data and normalize images'''
        batch_x = x_train_filenames[
                  batch_index * LearningConfig.batch_size:(batch_index + 1) * LearningConfig.batch_size]
        batch_y = y_train_filenames[
                  batch_index * LearningConfig.batch_size:(batch_index + 1) * LearningConfig.batch_size]
        img_batch = np.array([imread(img_path + file_name) for file_name in batch_x]) / 255.0
        if self.dataset_name == DatasetName.cofw:  # this ds is not normalized
            pn_batch = np.array([load(pn_tr_path + file_name) for file_name in batch_y])
        else:
            pn_batch = np.array([self._load_and_normalize(pn_tr_path + file_name) for file_name in batch_y])
        '''prediction to create tough and tolerant batches'''
        pn_batch_tough = model_tough_t.predict_on_batch(img_batch)
        if not train_dif:
            pn_batch_tol = model_tol_t.predict_on_batch(img_batch)
            pn_batch_stu = None
        else:
            pn_batch_tol = None
            pn_batch_stu = model_tol_t.predict_on_batch(img_batch)


        # pn_batch_tough = 0
        # pn_batch_tol = 0

        '''test: print'''
        # image_utility = ImageUtility()
        # if self.dataset_name == DatasetName.cofw:  # this ds is not normalized
        #     gr_s, gr_px_1, gr_Py_1 = image_utility.create_landmarks(pn_batch[0])
        #     tou_s, tou_px_1, tou_Py_1 = image_utility.create_landmarks(pn_batch_tough[0])
        #     tol_s, tol_px_1, tol_Py_1 = image_utility.create_landmarks(pn_batch_tol[0])
        # else:
        #     gr_s, gr_px_1, gr_Py_1 = image_utility.create_landmarks_from_normalized(pn_batch[0], 224, 224, 112, 112)
        #     tou_s, tou_px_1, tou_Py_1 = image_utility.create_landmarks_from_normalized(pn_batch_tough[0], 224, 224, 112, 112)
        #     tol_s, tol_px_1, tol_Py_1 = image_utility.create_landmarks_from_normalized(pn_batch_tol[0], 224, 224, 112, 112)
        #
        # imgpr.print_image_arr(str(batch_index)+'pts_gt', img_batch[0], gr_px_1, gr_Py_1)
        # imgpr.print_image_arr(str(batch_index)+'pts_t100', img_batch[0], tou_px_1, tou_Py_1)
        # imgpr.print_image_arr(str(batch_index)+'pts_t90', img_batch[0], tol_px_1, tol_Py_1)

        return img_batch, pn_batch, pn_batch_tough, pn_batch_tol, pn_batch_stu

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
