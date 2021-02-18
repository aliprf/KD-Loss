import math

import numpy as np
import tensorflow as tf

# tf.compat.v1.disable_eager_execution()
# tf.compat.v1.enable_eager_execution()

from PIL import Image
from tensorflow.keras import backend as K
from scipy.spatial import distance

from cnn_model import CNNModel
from configuration import DatasetName, D300wConf, LearningConfig
from image_utility import ImageUtility
from pca_utility import PCAUtility

print(tf.__version__)


class Custom_losses:
    def kd_loss(self, x_pr, x_gt, x_tough, x_tol, alpha_tough, alpha_mi_tough, alpha_tol, alpha_mi_tol,
                main_loss_weight, tough_loss_weight, tol_loss_weight):
        """km"""
        '''los KD'''
        # we revise teachers for reflection:
        x_tough = x_gt + tf.sign(x_pr - x_gt) * tf.abs(x_tough - x_gt)
        b_tough = x_gt + tf.sign(x_pr - x_gt) * tf.abs(x_tough - x_gt) * 0.15
        x_tol = x_gt + tf.sign(x_pr - x_gt) * tf.abs(x_tol - x_gt)
        b_tol = x_gt + tf.sign(x_pr - x_gt) * tf.abs(x_tol - x_gt) * 0.15
        # Region A: from T -> +inf
        tou_pos_map = tf.where(tf.sign(x_pr - x_tough) * tf.sign(x_tough - x_gt) > 0, alpha_tough, 0.0)
        tou_neg_map = tf.where(tf.sign(x_tough - x_pr) * tf.sign(x_pr - b_tough) >= 0, alpha_mi_tough, 0.0)
        # tou_red_map = tf.where(tf.sign(tf.abs(b_tough) - tf.abs(x_pr))*tf.sign(tf.abs(x_pr) - tf.abs(x_gt)) > 0, 0.1, 0.0)
        tou_map = tou_pos_map + tou_neg_map  # + tou_red_map

        tol_pos_map = tf.where(tf.sign(x_pr - x_tol) * tf.sign(x_tol - x_gt) > 0, alpha_tol, 0.0)
        tol_neg_map = tf.where(tf.sign(x_tol - x_pr) * tf.sign(x_pr - b_tol) >= 0, alpha_mi_tol, 0.0)
        # tol_red_map = tf.where(tf.sign(tf.abs(b_tol) - tf.abs(x_pr))*tf.sign(tf.abs(x_pr) - tf.abs(x_gt)) > 0, 0.1, 0.0)
        tol_map = tol_pos_map + tol_neg_map  # + tou_red_map

        '''calculate dif map for linear and non-linear part'''
        low_diff_main_map = tf.where(tf.abs(x_gt - x_pr) <= tf.abs(x_gt - x_tol), 1.0, 0.0)
        high_diff_main_map = tf.where(tf.abs(x_gt - x_pr) > tf.abs(x_gt - x_tol), 1.0, 0.0)

        '''calculate loss'''
        loss_main_high_dif = tf.reduce_mean(
            high_diff_main_map * (tf.square(x_gt - x_pr) + (3 * tf.abs(x_gt - x_tol)) - tf.square(x_gt - x_tol)))
        loss_main_low_dif = tf.reduce_mean(low_diff_main_map * (3 * tf.abs(x_gt - x_pr)))
        loss_main = main_loss_weight * (loss_main_high_dif + loss_main_low_dif)

        loss_tough_assist = tough_loss_weight * tf.reduce_mean(tou_map * tf.abs(x_tough - x_pr))
        loss_tol_assist = tol_loss_weight * tf.reduce_mean(tol_map * tf.abs(x_tol - x_pr))

        '''dif loss:'''
        loss_total = loss_main + loss_tough_assist + loss_tol_assist

        return loss_total, loss_main, loss_tough_assist, loss_tol_assist