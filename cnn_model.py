from configuration import DatasetName, DatasetType, \
    D300wConf, InputDataSize, LearningConfig

import tensorflow as tf
# tf.compat.v1.disable_eager_execution()

from tensorflow import keras
from skimage.transform import resize
from keras.regularizers import l2

# tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import mobilenet_v2, mobilenet, resnet50, densenet

from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, \
    BatchNormalization, Activation, GlobalAveragePooling2D, DepthwiseConv2D, Dropout, ReLU, Concatenate, Input

from keras.callbacks import ModelCheckpoint
from keras import backend as K

from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import math
from keras.callbacks import CSVLogger
from datetime import datetime

import cv2
import os.path
from keras.utils.vis_utils import plot_model
from scipy.spatial import distance
import scipy.io as sio

import efficientnet.tfkeras as efn


class CNNModel:
    def get_model(self, arch, input_tensor, output_len,
                  inp_shape=[InputDataSize.image_input_size, InputDataSize.image_input_size, 3], weight_path=None):
        if arch == 'efficientNet':
            model = self.create_efficientNet(inp_shape=inp_shape, input_tensor=input_tensor, output_len=output_len)
        elif arch == 'mobileNetV2':
            model = self.create_MobileNet(inp_shape=inp_shape, inp_tensor=input_tensor, output_len=output_len,
                                          weight_path=weight_path)
        elif arch == 'mobileNetV2_d':
            model = self.create_MobileNet_with_drop(inp_shape=inp_shape, inp_tensor=input_tensor,
                                                    output_len=output_len, weight_path=weight_path)
        return model

    def create_MobileNet_with_drop(self, inp_shape, inp_tensor, output_len, weight_path):
        initializer = tf.keras.initializers.glorot_uniform()
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=inp_shape, alpha=1.0, include_top=True, weights=None,
                                                   input_tensor=inp_tensor, pooling=None)
        mobilenet_model.layers.pop()
        x = mobilenet_model.get_layer('global_average_pooling2d').output  # 1280
        x = Dense(output_len, name='O_L')(x)
        inp = mobilenet_model.input
        model = Model(inp, x)
        model.load_weights(weight_path)
        model.summary()

        '''revise model and add droput'''
        model.layers.pop()
        x = model.get_layer('global_average_pooling2d').output  # 1280
        x = Dropout(0.5)(x)
        out_landmarks = Dense(output_len, activation=keras.activations.linear,
                              use_bias=False, kernel_initializer=initializer, name='O_L')(x)
        inp = mobilenet_model.input
        revised_model = Model(inp, out_landmarks)
        revised_model.summary()
        revised_model.save_weights('W_ds_wflw_mn_base_with_drop.h5')
        revised_model.save('M_ds_wflw_mn_base_with_drop.h5')
        model_json = revised_model.to_json()
        with open("mobileNet_v2_main.json", "w") as json_file:
            json_file.write(model_json)
        return revised_model

    def create_MobileNet_dif(self, inp_shape, inp_tensor, output_len, is_old, weight_path):
        initializer = tf.keras.initializers.glorot_uniform()

        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=inp_shape,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=inp_tensor,
                                                   pooling=None)
        mobilenet_model.layers.pop()
        # mobilenet_model.summary()

        # global_avg = mobilenet_model.get_layer('global_average_pooling2d_2').output  # 1280
        global_avg = mobilenet_model.get_layer('global_average_pooling2d').output  # 1280
        x = Dense(3 * output_len)(global_avg)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dense(2 * output_len)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(0.3)(x)
        dif_gt_st = Dense(output_len, name='dif_gt_st')(x)

        x = Dense(3 * output_len)(global_avg)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dense(2 * output_len)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(0.3)(x)
        dif_gt_pt = Dense(output_len, name='dif_gt_pt')(x)

        '''now we add other layers'''
        # global_avg = revised_model.get_layer('global_average_pooling2d').output  # 1280
        inp = mobilenet_model.input
        revised_model = Model(inp, [dif_gt_st, dif_gt_pt])
        model_json = revised_model.to_json()
        # revised_model.save('ds_300w_stu_.h5')
        with open("mobileNet_v2_stu_dif.json", "w") as json_file:
            json_file.write(model_json)
        return revised_model

    def create_MobileNet(self, inp_shape, inp_tensor, output_len, is_old, weight_path):
        # initializer = tf.keras.initializers.HeUniform()
        initializer = tf.keras.initializers.glorot_uniform()

        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=inp_shape,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=inp_tensor,
                                                   pooling=None)

        # global_avg = revised_model.get_layer('global_average_pooling2d').output  # 1280
        inp = mobilenet_model.input
        out_landmarks = mobilenet_model.get_layer('O_L').output
        revised_model = Model(inp, [out_landmarks])
        model_json = revised_model.to_json()
        # revised_model.save('ds_300w_stu_.h5')
        with open("mobileNet_v2_stu.json", "w") as json_file:
            json_file.write(model_json)
        return revised_model

        # x = Dense(3*output_len)(global_avg)
        # x = BatchNormalization()(x)
        # x = ReLU()(x)
        # x = Dense(2 * output_len)(x)
        # x = BatchNormalization()(x)
        # x = ReLU()(x)
        # x = Dropout(0.1)(x)
        # out_tol = Dense(output_len, name='O_tol')(x)
        # out_tol_dif_gt = Dense(output_len, name='O_tol_d_g')(x)
        #
        # x = Dense(3*output_len)(global_avg)
        # x = BatchNormalization()(x)
        # x = ReLU()(x)
        # x = Dense(2 * output_len)(x)
        # x = BatchNormalization()(x)
        # x = ReLU()(x)
        # x = Dropout(0.3)(x)
        # out_tou = Dense(output_len, name='O_tou')(x)
        # out_tou_dif_gt = Dense(output_len, name='O_tou_d_g')(x)
        #
        # revised_model = Model(inp, [out_landmarks, out_tol, out_tol_dif_gt, out_tou, out_tou_dif_gt])
        # model_json = revised_model.to_json()
        # revised_model.save('ds_300w_3o_stu_.h5')
        # with open("mobileNet_v2_5_out.json", "w") as json_file:
        #     json_file.write(model_json)
        #
        # return revised_model

    def hour_glass_network(self, num_classes=68, num_stacks=10, num_filters=256,
                           in_shape=(224, 224), out_shape=(56, 56)):
        hg_net = HourglassNet(num_classes=num_classes, num_stacks=num_stacks,
                              num_filters=num_filters,
                              in_shape=in_shape,
                              out_shape=out_shape)
        model = hg_net.build_model(mobile=True)
        return model

    def mn_asm_v0(self, tensor):
        """
            has only one output
            we use custom loss for this network and using ASM to correct points after that
        """

        # block_13_project_BN block_10_project_BN block_6_project_BN
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=None,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=tensor,
                                                   pooling=None)

        mobilenet_model.layers.pop()

        inp = mobilenet_model.input

        '''heatmap can not be generated from activation layers, so we use out_relu'''
        x = mobilenet_model.get_layer('out_relu').output  # 7, 7, 1280
        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='deconv1', kernel_initializer='he_uniform')(x)  # 14, 14, 256
        x = BatchNormalization(name='out_bn1')(x)

        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='deconv2', kernel_initializer='he_uniform')(x)  # 28, 28, 256
        x = BatchNormalization(name='out_bn2')(x)

        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='deconv3', kernel_initializer='he_uniform')(x)  # 56, 56, 256
        x = BatchNormalization(name='out_bn3')(x)

        out_heatmap = Conv2D(LearningConfig.point_len, kernel_size=1, padding='same', name='out_heatmap')(x)

        revised_model = Model(inp, out_heatmap)

        revised_model.summary()
        # plot_model(revised_model, to_file='mnv2_hm.png', show_shapes=True, show_layer_names=True)
        model_json = revised_model.to_json()

        with open("mn_asm_v0.json", "w") as json_file:
            json_file.write(model_json)

        return revised_model

    def mn_asm_v1(self, tensor):
        # block_13_project_BN block_10_project_BN block_6_project_BN
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=None,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=tensor,
                                                   pooling=None)

        mobilenet_model.layers.pop()

        inp = mobilenet_model.input

        # '''block_1 {  block_6_project_BN 14, 14, 46 '''
        # x = mobilenet_model.get_layer('block_6_project_BN').output  # 14, 14, 46
        # x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
        #                     name='block_1_deconv_1', kernel_initializer='he_uniform')(x)  # 28, 28, 128
        # x = BatchNormalization(name='block_1_out_bn_1')(x)
        #
        # x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
        #                     name='block_1_deconv_2', kernel_initializer='he_uniform')(x)  # 56, 56, 128
        # x = BatchNormalization(name='block_1_out_bn_2')(x)
        #
        # block_1_out = Conv2D(LearningConfig.landmark_len // 2, kernel_size=1, padding='same', name='block_1_out')(x)
        # '''block_1 }'''

        '''block_2 {  block_10_project_BN 14, 14, 96 '''
        x = mobilenet_model.get_layer('block_10_project_BN').output  # 14, 14, 96
        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='block_2_deconv_1', kernel_initializer='he_uniform')(x)  # 28, 28, 128
        x = BatchNormalization(name='block_2_out_bn_1')(x)

        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='block_2_deconv_2', kernel_initializer='he_uniform')(x)  # 56, 56, 128
        x = BatchNormalization(name='block_2_out_bn_2')(x)

        block_2_out = Conv2D(LearningConfig.landmark_len // 2, kernel_size=1, padding='same', name='block_2_out')(x)
        '''block_2 }'''

        '''block_3 {  block_13_project_BN 7, 7, 160 '''
        x = mobilenet_model.get_layer('block_13_project_BN').output  # 7, 7, 160
        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='block_3_deconv_1', kernel_initializer='he_uniform')(x)  # 14, 14, 128
        x = BatchNormalization(name='block_3_out_bn_1')(x)

        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='block_3_deconv_2', kernel_initializer='he_uniform')(x)  # 28, 28, 128
        x = BatchNormalization(name='block_3_out_bn_2')(x)

        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='block_3_deconv_3', kernel_initializer='he_uniform')(x)  # 56, 56, 128
        x = BatchNormalization(name='block_3_out_bn_3')(x)

        block_3_out = Conv2D(LearningConfig.landmark_len // 2, kernel_size=1, padding='same', name='block_3_out')(x)

        '''block_3 }'''

        '''heatmap can not be generated from activation layers, so we use out_relu'''
        x = mobilenet_model.get_layer('out_relu').output  # 7, 7, 1280
        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='deconv1', kernel_initializer='he_uniform')(x)  # 14, 14, 256
        x = BatchNormalization(name='out_bn1')(x)

        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='deconv2', kernel_initializer='he_uniform')(x)  # 28, 28, 256
        x = BatchNormalization(name='out_bn2')(x)

        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='deconv3', kernel_initializer='he_uniform')(x)  # 56, 56, 256
        x = BatchNormalization(name='out_bn3')(x)

        out_heatmap = Conv2D(LearningConfig.landmark_len // 2, kernel_size=1, padding='same', name='out_heatmap')(x)

        revised_model = Model(inp, [
            # block_1_out,  # 85
            # block_2_out,  # 90
            # block_3_out,  # 97
            out_heatmap  # 100
        ])

        revised_model.summary()
        # plot_model(revised_model, to_file='mnv2_hm.png', show_shapes=True, show_layer_names=True)
        model_json = revised_model.to_json()

        with open("mn_asm_v1.json", "w") as json_file:
            json_file.write(model_json)

        return revised_model

    def mnv2_hm_r_v2(self, inp_shape):
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=inp_shape,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=None,
                                                   pooling=None)

        mobilenet_model.layers.pop()

        inp = mobilenet_model.input
        '''heatmap can not be generated from activation layers, so we use out_relu'''
        x = mobilenet_model.get_layer('out_relu').output  # 7, 7, 1280
        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same',
                            kernel_initializer='he_uniform')(x)  # 14, 14, 256
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same',
                            kernel_initializer='he_uniform')(x)  # 28, 28, 256
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same',
                            kernel_initializer='he_uniform')(x)  # 56, 56, 256
        bn_0 = BatchNormalization(name='bn_0')(x)
        x = ReLU()(bn_0)

        '''reduce to  7'''
        x = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer='he_uniform')(
            x)  # 28, 28, 256
        bn_1 = BatchNormalization(name='bn_1')(x)  # 28, 28, 256
        x = ReLU()(bn_1)

        x = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer='he_uniform')(
            x)  # 14, 14, 256
        bn_2 = BatchNormalization(name='bn_2')(x)  # 14, 14, 256
        x = ReLU()(bn_2)

        x = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer='he_uniform')(
            x)  # 7, 7 , 256
        bn_3 = BatchNormalization(name='bn_3')(x)  # 7, 7 , 256
        x = ReLU()(bn_3)

        '''increase to  56'''
        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same',
                            name='deconv1', kernel_initializer='he_uniform')(x)  # 14, 14, 256
        x = BatchNormalization()(x)
        x = keras.layers.add([x, bn_2])  # 14, 14, 256
        x = ReLU()(x)

        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same',
                            name='deconv2', kernel_initializer='he_uniform')(x)  # 28, 28, 256
        x = BatchNormalization()(x)
        x = keras.layers.add([x, bn_1])  # 28, 28, 256
        x = ReLU()(x)

        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same',
                            name='deconv3', kernel_initializer='he_uniform')(x)  # 56, 56, 256
        x = BatchNormalization()(x)
        x = keras.layers.add([x, bn_0])  # 56, 56, 256

        '''out_conv_layer'''
        out_heatmap = Conv2D(LearningConfig.landmark_len // 2, kernel_size=1, padding='same', name='out_heatmap')(x)

        revised_model = Model(inp, [
            out_heatmap,
        ])

        revised_model.summary()
        # plot_model(revised_model, to_file='mnv2_hm.png', show_shapes=True, show_layer_names=True)
        model_json = revised_model.to_json()

        with open("mnv2_hm_r_v2.json", "w") as json_file:
            json_file.write(model_json)

        return revised_model

    # def create_multi_branch_mn(self, inp_shape, num_branches):
    #     branches = []
    #     inputs = []
    #     for i in range(num_branches):
    #         inp_i, br_i = self.create_branch_mn(prefix=str(i), inp_shape=inp_shape)
    #         inputs.append(inp_i)
    #         branches.append(br_i)
    #
    #     revised_model = Model(inputs[0], branches[0], name='multiBranchMN')
    #     # revised_model = Model(inputs, branches, name='multiBranchMN')
    #
    #     revised_model.layers.pop(0)
    #
    #     new_input = Input(shape=inp_shape)
    #
    #     revised_model = Model(new_input, revised_model.outputs)
    #
    #     revised_model.summary()
    #
    #     model_json = revised_model.to_json()
    #     with open("MultiBranchMN.json", "w") as json_file:
    #         json_file.write(model_json)
    #     return revised_model
    #

    def create_multi_branch_mn(self, inp_shape, num_branches):

        mobilenet_model = mobilenet_v2.MobileNetV2_mb(3, input_shape=inp_shape,
                                                      alpha=1.0,
                                                      include_top=True,
                                                      weights=None,
                                                      input_tensor=None,
                                                      pooling=None)

        mobilenet_model.layers.pop()
        inp = mobilenet_model.input

        outputs = []
        for i in range(num_branches):
            prefix = str(i)
            '''heatmap can not be generated from activation layers, so we use out_relu'''
            x = mobilenet_model.get_layer('out_relu' + prefix).output  # 7, 7, 1280

            x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                                name=prefix + '_deconv1', kernel_initializer='he_uniform')(x)  # 14, 14, 256
            x = BatchNormalization(name=prefix + 'out_bn1')(x)

            x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                                name=prefix + '_deconv2', kernel_initializer='he_uniform')(x)  # 28, 28, 256
            x = BatchNormalization(name=prefix + 'out_bn2')(x)

            x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                                name=prefix + '_deconv3', kernel_initializer='he_uniform')(x)  # 56, 56, 256
            x = BatchNormalization(name=prefix + 'out_bn3')(x)

            out_heatmap = Conv2D(LearningConfig.point_len, kernel_size=1, padding='same', name=prefix + '_out_hm')(x)
            outputs.append(out_heatmap)

        revised_model = Model(inp, outputs)

        revised_model.summary()

        model_json = revised_model.to_json()
        with open("MultiBranchMN.json", "w") as json_file:
            json_file.write(model_json)
        return revised_model

    def create_branch_mn(self, prefix, inp_shape):
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=inp_shape,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=None,
                                                   pooling=None)
        mobilenet_model.layers.pop()
        inp = mobilenet_model.input

        '''heatmap can not be generated from activation layers, so we use out_relu'''
        x = mobilenet_model.get_layer('out_relu').output  # 7, 7, 1280

        x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name=prefix + '_deconv1', kernel_initializer='he_uniform')(x)  # 14, 14, 256
        x = BatchNormalization(name=prefix + 'out_bn1')(x)

        x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name=prefix + '_deconv2', kernel_initializer='he_uniform')(x)  # 28, 28, 256
        x = BatchNormalization(name=prefix + 'out_bn2')(x)

        x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name=prefix + '_deconv3', kernel_initializer='he_uniform')(x)  # 56, 56, 256
        x = BatchNormalization(name=prefix + 'out_bn3')(x)

        out_heatmap = Conv2D(LearningConfig.point_len, kernel_size=1, padding='same', name=prefix + '_out_hm')(x)

        for layer in mobilenet_model.layers:
            layer.name = layer.name + '_' + prefix
        return inp, out_heatmap

    # def create_multi_branch_mn_one_input(self, inp_shape, num_branches):
    #     mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=inp_shape,
    #                                                alpha=1.0,
    #                                                include_top=True,
    #                                                weights=None,
    #                                                input_tensor=None,
    #                                                pooling=None)
    #     mobilenet_model.layers.pop()
    #     inp = mobilenet_model.input
    #     outputs = []
    #     relu_name = 'out_relu'
    #     for i in range(num_branches):
    #         x = mobilenet_model.get_layer(relu_name).output  # 7, 7, 1280
    #         prefix = str(i)
    #         for layer in mobilenet_model.layers:
    #             layer.name = layer.name + prefix
    #
    #         relu_name = relu_name + prefix
    #
    #         '''heatmap can not be generated from activation layers, so we use out_relu'''
    #
    #         x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
    #                             name=prefix+'_deconv1', kernel_initializer='he_uniform')(x)  # 14, 14, 256
    #         x = BatchNormalization(name=prefix + 'out_bn1')(x)
    #
    #         x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
    #                             name=prefix+'_deconv2', kernel_initializer='he_uniform')(x)  # 28, 28, 256
    #         x = BatchNormalization(name=prefix +'out_bn2')(x)
    #
    #         x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
    #                             name=prefix+'_deconv3', kernel_initializer='he_uniform')(x)  # 56, 56, 256
    #         x = BatchNormalization(name=prefix+'out_bn3')(x)
    #
    #         out_heatmap = Conv2D(LearningConfig.point_len, kernel_size=1, padding='same', name=prefix+'_out_hm')(x)
    #         outputs.append(out_heatmap)
    #
    #     revised_model = Model(inp, outputs)
    #
    #     revised_model.summary()
    #
    #     model_json = revised_model.to_json()
    #     with open("MultiBranchMN.json", "w") as json_file:
    #         json_file.write(model_json)
    #     return revised_model

    def create_efficientNet(self, inp_shape, input_tensor, output_len, is_teacher=True):
        if is_teacher:  # for teacher we use a heavier network
            eff_net = efn.EfficientNetB3(include_top=True,
                                         weights=None,
                                         input_tensor=None,
                                         input_shape=[InputDataSize.image_input_size, InputDataSize.image_input_size,
                                                      3],
                                         pooling=None,
                                         classes=output_len)
            # return self._create_efficientNet_3deconv(inp_shape, input_tensor, output_len)
        else:  # for student we use the small network
            eff_net = efn.EfficientNetB0(include_top=True,
                                         weights=None,
                                         input_tensor=None,
                                         input_shape=inp_shape,
                                         pooling=None,
                                         classes=output_len)  # or weights='noisy-student'

        eff_net.layers.pop()
        inp = eff_net.input

        x = eff_net.get_layer('top_activation').output
        x = GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(rate=0.5)(x)
        output = Dense(output_len, activation='linear', name='out')(x)

        eff_net = Model(inp, output)

        eff_net.summary()

        # plot_model(eff_net, to_file='eff_net.png', show_shapes=True, show_layer_names=True)

        # tf.keras.utils.plot_model(
        #     eff_net,
        #     to_file="eff_net.png",
        #     show_shapes=False,
        #     show_layer_names=True,
        #     rankdir="TB"
        # )

        # model_json = eff_net.to_json()
        # with open("eff_net.json", "w") as json_file:
        #     json_file.write(model_json)
        return eff_net

    def _create_efficientNet_3deconv(self, inp_shape, input_tensor, output_len):
        eff_net = efn.EfficientNetB5(include_top=True,
                                     weights=None,
                                     input_tensor=input_tensor,
                                     input_shape=[224, 224, 3],
                                     pooling=None,
                                     classes=output_len)
        eff_net.layers.pop()
        inp = eff_net.input

        x = eff_net.get_layer('top_bn').output

        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu',
                            kernel_initializer='he_uniform')(x)  # 14, 14, 256
        x = BatchNormalization()(x)

        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu',
                            kernel_initializer='he_uniform')(x)  # 28, 28, 256
        x = BatchNormalization()(x)

        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu',
                            kernel_initializer='he_uniform')(x)  # 56, 56, 256
        x = BatchNormalization()(x)

        out_heatmap = Conv2D(output_len // 2, kernel_size=1, padding='same')(x)

        eff_net = Model(inp, out_heatmap)

        eff_net.summary()

        return eff_net

    def create_asmnet(self, inp_shape, num_branches, output_len):
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=inp_shape,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=None,
                                                   pooling=None)

        mobilenet_model.layers.pop()
        inp = mobilenet_model.input
        outputs = []
        relu_name = 'out_relu'
        for i in range(num_branches):
            x = mobilenet_model.get_layer(relu_name).output  # 7, 7, 1280
            prefix = str(i)
            for layer in mobilenet_model.layers:
                layer.name = layer.name + prefix

            relu_name = relu_name + prefix

            '''heatmap can not be generated from activation layers, so we use out_relu'''

            x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                                name=prefix + '_deconv1', kernel_initializer='he_uniform')(x)  # 14, 14, 256
            x = BatchNormalization(name=prefix + 'out_bn1')(x)

            x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                                name=prefix + '_deconv2', kernel_initializer='he_uniform')(x)  # 28, 28, 256
            x = BatchNormalization(name=prefix + 'out_bn2')(x)

            x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                                name=prefix + '_deconv3', kernel_initializer='he_uniform')(x)  # 56, 56, 256
            x = BatchNormalization(name=prefix + 'out_bn3')(x)

            out_heatmap = Conv2D(output_len, kernel_size=1, padding='same', name=prefix + '_out_hm')(x)
            outputs.append(out_heatmap)

        revised_model = Model(inp, outputs)

        revised_model.summary()

        model_json = revised_model.to_json()
        with open("asmnet.json", "w") as json_file:
            json_file.write(model_json)
        return revised_model

    def mnv2_hm(self, tensor):
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=None,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=tensor,
                                                   pooling=None)

        mobilenet_model.layers.pop()

        inp = mobilenet_model.input
        '''heatmap can not be generated from activation layers, so we use out_relu'''
        x = mobilenet_model.get_layer('out_relu').output  # 7, 7, 1280
        x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='deconv1', kernel_initializer='he_uniform')(x)  # 14, 14, 256
        x = BatchNormalization(name='out_bn1')(x)

        x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='deconv2', kernel_initializer='he_uniform')(x)  # 28, 28, 256
        x = BatchNormalization(name='out_bn2')(x)

        x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='deconv3', kernel_initializer='he_uniform')(x)  # 56, 56, 256
        x = BatchNormalization(name='out_bn3')(x)

        out_heatmap = Conv2D(LearningConfig.landmark_len // 2, kernel_size=1, padding='same', name='out_heatmap')(x)

        revised_model = Model(inp, [
            out_heatmap,
        ])

        revised_model.summary()
        # plot_model(revised_model, to_file='mnv2_hm.png', show_shapes=True, show_layer_names=True)
        model_json = revised_model.to_json()

        with open("mnv2_hm.json", "w") as json_file:
            json_file.write(model_json)

        return revised_model

    def mobileNet_v2_main_discriminator(self, tensor, input_shape):
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=input_shape,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=tensor,
                                                   pooling=None)
        # , classes=cnf.landmark_len)

        mobilenet_model.layers.pop()

        x = mobilenet_model.get_layer('global_average_pooling2d_2').output  # 1280
        softmax = Dense(1, activation='sigmoid', name='out')(x)
        inp = mobilenet_model.input

        revised_model = Model(inp, softmax)

        revised_model.summary()
        # plot_model(revised_model, to_file='mobileNet_v2_main.png', show_shapes=True, show_layer_names=True)
        model_json = revised_model.to_json()

        with open("mobileNet_v2_main.json", "w") as json_file:
            json_file.write(model_json)

        return revised_model

    def mobileNet_v2_main(self, tensor):
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=None,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=tensor,
                                                   pooling=None)
        # , classes=cnf.landmark_len)

        mobilenet_model.layers.pop()

        x = mobilenet_model.get_layer('global_average_pooling2d_1').output  # 1280
        x = Dense(LearningConfig.landmark_len, name='dense_layer_out_2', activation='relu',
                  kernel_initializer='he_uniform')(x)
        Logits = Dense(LearningConfig.landmark_len, name='out')(x)
        inp = mobilenet_model.input

        revised_model = Model(inp, Logits)

        revised_model.summary()
        # plot_model(revised_model, to_file='mobileNet_v2_main.png', show_shapes=True, show_layer_names=True)
        model_json = revised_model.to_json()

        with open("mobileNet_v2_main.json", "w") as json_file:
            json_file.write(model_json)

        return revised_model
