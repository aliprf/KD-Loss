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

        if dataset_name == DatasetName.ibug:
            self.output_len = D300wConf.num_of_landmarks * 2
        elif dataset_name == DatasetName.cofw_test:
            self.output_len = CofwConf.num_of_landmarks * 2
        elif dataset_name == DatasetName.wflw_test:
            self.output_len = WflwConf.num_of_landmarks * 2

        cnn = CNNModel()
        model = cnn.get_model(train_images=None, arch=arch, num_output_layers=num_output_layers,
                              input_tensor=None, output_len=self.output_len)

        model.load_weights(weight_fname)

        if dataset_name == DatasetName.ibug:
            self._test_on_W300(model)
        elif dataset_name == DatasetName.cofw_test:
            self._test_on_COFW(model)
        elif dataset_name == DatasetName.wflw_test:
            self._test_on_COFW(model)

    def _test_on_WFLW(self, model):
        tf_record_utility = TFRecordUtility(self.output_len)
        lbl_arr_total, img_arr_total = tf_record_utility.retrieve_tf_record_test_set(
            tfrecord_filename=WflwConf.tf_test_path,
            number_of_records=WflwConf.orig_number_of_test,
            only_label=False)
        lbl_arr_total = np.array(lbl_arr_total)
        img_arr_total = np.array(img_arr_total)

        nme, fr, auc = self._calculate_errors(model, CofwConf.orig_number_of_test, img_arr_total, lbl_arr_total)
        print('nme: ', str(nme), 'fr: ', str(fr), 'auc: ', str(auc))
        print("-------------------------------------------------------------")

    def _test_on_COFW(self, model):
        tf_record_utility = TFRecordUtility(self.output_len)
        lbl_arr_total, img_arr_total = tf_record_utility.retrieve_tf_record_test_set(
            tfrecord_filename=CofwConf.tf_test_path,
            number_of_records=CofwConf.orig_number_of_test,
            only_label=False)
        lbl_arr_total = np.array(lbl_arr_total)
        img_arr_total = np.array(img_arr_total)

        nme, fr, auc, = self._calculate_errors(model, CofwConf.orig_number_of_test, img_arr_total, lbl_arr_total)
        print('nme: ', str(nme), 'fr: ', str(fr), 'auc: ', str(auc))
        print("-------------------------------------------------------------")

    def _test_on_W300(self, model):
        tf_record_utility = TFRecordUtility(self.output_len)
        lbl_arr_challenging, img_arr_challenging = tf_record_utility.retrieve_tf_record_test_set(
            tfrecord_filename=W300Conf.tf_challenging,
            number_of_records=W300Conf.number_of_all_sample_challenging,
            only_label=False)
        lbl_arr_common, img_arr_common = tf_record_utility.retrieve_tf_record_test_set(
            tfrecord_filename=W300Conf.tf_common,
            number_of_records=W300Conf.number_of_all_sample_common,
            only_label=False)
        lbl_arr_full, img_arr_full = tf_record_utility.retrieve_tf_record_test_set(
            tfrecord_filename=W300Conf.tf_full,
            number_of_records=W300Conf.number_of_all_sample_full,
            only_label=False)

        lbl_arr_challenging = np.array(lbl_arr_challenging)
        img_arr_challenging = np.array(img_arr_challenging)

        lbl_arr_common = np.array(lbl_arr_common)
        img_arr_common = np.array(img_arr_common)

        lbl_arr_full = np.array(lbl_arr_full)
        img_arr_full = np.array(img_arr_full)

        nme_ch, fr_ch, auc_ch = self._calculate_errors(
            model, W300Conf.number_of_all_sample_challenging,
            img_arr_challenging, lbl_arr_challenging)
        print('nme_ch: ', str(nme_ch), 'fr_ch: ', str(fr_ch), 'auc_ch: ', str(auc_ch))

        nme_c, fr_c, auc_c = self._calculate_errors(
            model,
            W300Conf.number_of_all_sample_common,
            img_arr_common, lbl_arr_common)
        print('nme_c: ', str(nme_c), 'fr_c: ', str(fr_c), 'auc_c: ', str(auc_c))

        nme_f, fr_f, auc_f = self._calculate_errors(model, W300Conf.number_of_all_sample_full, img_arr_full, lbl_arr_full)
        print('nme_f: ', str(nme_f), 'fr_f: ', str(fr_f), 'auc_f: ', str(auc_f))

    def _calculate_errors(self, model, number_test_set, test_img_arr, test_lbl_arr):
        fr_threshold = 0.1
        ACU = []

        fail_counter = 0
        sum_loss = 0
        all_true = []
        all_predicted = []

        for i in tqdm(range(number_test_set)):
            loss, lt, lp = self._test_result_per_image(i, model, test_img_arr[i], test_lbl_arr[i])
            sum_loss += loss

            if loss > fr_threshold:
                fail_counter += 1

            all_true.append(lt)
            all_predicted.append(lp)

        sio.savemat('all_true.mat', {'ground_truth_all': np.array(all_true)})
        sio.savemat('all_pridicted.mat', {'detected_points_all': np.array(all_predicted)})

        nme = sum_loss * 100 / number_test_set

        fr = 100 * fail_counter / number_test_set
        return nme, fr, ACU

    def _test_result_per_image(self, counter, model, img, labels_true):
        image_utility = ImageUtility()
        image = np.expand_dims(img, axis=0)
        predict = model.predict(image)

        pre_points = predict[0][0]
        pose_predicted = predict[1][0]

        labels_true_transformed, landmark_arr_x_t, landmark_arr_y_t = image_utility. \
            create_landmarks_from_normalized(labels_true, 224, 224, 112, 112)

        labels_predict_transformed, landmark_arr_x_p, landmark_arr_y_p = \
            image_utility.create_landmarks_from_normalized(pre_points, 224, 224, 112, 112)

        # auc = self.calculate_AUC(labels_true, pre_points)

        '''asm pp'''
        # xy_h_p_asm = self.__post_process_correction(xy_h_p, True)
        # labels_predict_transformed_asm, landmark_arr_x_p_asm, landmark_arr_y_p_asm = image_utility. \
        #     create_landmarks_from_normalized(xy_h_p_asm, 224, 224, 112, 112)
        # labels_predict_transformed = labels_predict_transformed_asm
        ''''''

        '''test print'''
        # imgpr.print_image_arr((counter+1)*1000, img, landmark_arr_x_p, landmark_arr_y_p)
        # imgpr.print_image_arr((counter+1), img, landmark_arr_x_t, landmark_arr_y_t)

        # print("landmark_arr_x_t: " + str(landmark_arr_x_t))
        # print("landmark_arr_x_p :" + str(landmark_arr_x_p))
        #
        # print("landmark_arr_y_t: " + str(landmark_arr_y_t))
        # print("landmark_arr_y_p :" + str(landmark_arr_y_p))

        # return 0, 0, 0, 0, 0, 0

        # interpupil_distance = self.__calculate_interpupil_distance(labels_true_transformed)
        interpupil_distance = self.__calculate_interoccular_distance(labels_true_transformed)

        sum_errors = 0
        for i in range(0, len(labels_true_transformed), 2):  # two step each time
            '''this is the value after transformation to the real points'''
            x_point_predicted = labels_predict_transformed[i]
            y_point_predicted = labels_predict_transformed[i + 1]

            # x_point_predicted_asm = labels_predict_asm_transformed[i]
            # y_point_predicted_asm = labels_predict_asm_transformed[i+1]
            #

            x_point_true = labels_true_transformed[i]
            y_point_true = labels_true_transformed[i + 1]

            '''this is the normalized value, which predicted by network'''
            error = math.sqrt(((x_point_predicted - x_point_true) ** 2) + ((y_point_predicted - y_point_true) ** 2))
            sum_errors += error

        normalized_mean_error = sum_errors / (interpupil_distance * (self.output_len / 2))
        # print(normalized_mean_error)
        # print('=====')

        lp = np.array(labels_predict_transformed).reshape([self.output_len // 2, 2])
        lt = np.array(labels_true_transformed).reshape([self.output_len // 2, 2])

        # print(labels_true_transformed)
        # print(lt)
        # print('---------------')

        return normalized_mean_error, lt, lp

    def __calculate_interoccular_distance(self, labels_true):
        if self.dataset_name == DatasetName.ibug:
            left_oc_x = labels_true[72]
            left_oc_y = labels_true[73]
            right_oc_x = labels_true[90]
            right_oc_y = labels_true[91]
        elif self.dataset_name == DatasetName.cofw_test:
            left_oc_x = labels_true[16]
            left_oc_y = labels_true[17]
            right_oc_x = labels_true[18]
            right_oc_y = labels_true[19]

        distance = math.sqrt(((left_oc_x - right_oc_x) ** 2) + ((left_oc_y - right_oc_y) ** 2))
        return distance

    def __calculate_interpupil_distance(self, labels_true):
        # points: x,y 36--> 41 point for left, and 42->47 for right

        left_pupil_x = (labels_true[72] + labels_true[74] + labels_true[76] + labels_true[78] + labels_true[80] +
                        labels_true[82]) / 6
        left_pupil_y = (labels_true[73] + labels_true[75] + labels_true[77] + labels_true[79] + labels_true[81] +
                        labels_true[83]) / 6

        right_pupil_x = (labels_true[84] + labels_true[86] + labels_true[88] + labels_true[90] + labels_true[92] +
                         labels_true[94]) / 6
        right_pupil_y = (labels_true[85] + labels_true[87] + labels_true[89] + labels_true[91] + labels_true[93] +
                         labels_true[95]) / 6

        dis = math.sqrt(((left_pupil_x - right_pupil_x) ** 2) + ((left_pupil_y - right_pupil_y) ** 2))

        # p1 = [left_pupil_x, left_pupil_y]
        # p2 = [right_pupil_x, right_pupil_y]
        # dis1 = distance.euclidean(p1, p2)
        #
        # print(dis)
        # print(dis1)
        # print('==============') both are equal
        return dis

    def test_per_image(self, counter, model, img, multitask, detect, org_img, x1, y1, scale_x, scale_y):

        image_utility = ImageUtility()
        pose_predicted = []
        image = np.expand_dims(img, axis=0)
        if multitask:
            labels_main = np.swapaxes(model.predict(image)[0], 0, 1)
            # noise_lbls = model.predict(image)[1]
            # eyes_lbls = model.predict(image)[2]
            # face_lbls = model.predict(image)[3]
            # mouth_lbls = model.predict(image)[4]
            pose_predicted = model.predict(image)[1][0]
            # lbls_partial = np.swapaxes(np.concatenate((face_lbls, noise_lbls, eyes_lbls, mouth_lbls), axis=1), 0, 1)  #

            labels_predicted = labels_main
            # labels_predicted = lbls_partial
            # labels_predicted = (lbls_partial + labels_main) / 2.0
        else:
            labels_predicted = np.swapaxes(model.predict(image), 0, 1)

        # labels_true_transformed, landmark_arr_x_t, landmark_arr_y_t = image_utility. \
        #     create_landmarks_from_normalized(labels_true, 224, 224, 112, 112)
        #
        # labels_predicted_asm = self.__post_process_correction(labels_predicted)

        # labels_predict_asm_transformed, landmark_arr_x_asm_p, landmark_arr_y_asm_p = image_utility. \
        #     create_landmarks_from_normalized(labels_predicted_asm, 224, 224, 112, 112)

        # imgpr.print_image_arr(counter+1, img, [], [])
        # imgpr.print_image_arr(counter+1, img, landmark_arr_x_p, landmark_arr_y_p)

        # imgpr.print_image_arr(counter+1, org_img, landmark_arr_x_p, landmark_arr_y_p)

        # imgpr.print_image_arr((counter+1)*1000, img, landmark_arr_x_t, landmark_arr_y_t)
        # imgpr.print_image_arr((counter+1)*100000, img, landmark_arr_x_asm_p, landmark_arr_y_asm_p)

        '''pose estimation vs hopeNet'''
        img_cp_1 = np.array(img) * 255.0
        r, g, b = cv2.split(img_cp_1)
        img_cp_1 = cv2.merge([b, g, r])

        img_cp_2 = np.array(img) * 255.0
        r, g, b = cv2.split(img_cp_2)
        img_cp_2 = cv2.merge([b, g, r])

        # yaw_truth, pitch_truth, roll_truth = 0, 0, 0
        yaw_truth, pitch_truth, roll_truth = detect.detect(img, isFile=False, show=False)

        yaw_p = pose_predicted[0]
        pitch_p = pose_predicted[1]
        roll_p = pose_predicted[2]
        ''' normalized to normal '''
        min_degree = - 65
        max_degree = 65
        yaw_tpre = min_degree + (max_degree - min_degree) * (yaw_p + 1) / 2
        pitch_tpre = min_degree + (max_degree - min_degree) * (pitch_p + 1) / 2
        roll_tpre = min_degree + (max_degree - min_degree) * (roll_p + 1) / 2

        output_pre = utils.draw_axis(org_img, yaw_tpre, pitch_tpre, roll_tpre, tdx=float(x1) + 112, tdy=float(y1) + 112,
                                     size=112)

        labels_predict_transformed, landmark_arr_x_p, landmark_arr_y_p, img_cv2 = image_utility. \
            create_landmarks_from_normalized_original_img(output_pre, labels_predicted, 224, 224, 112, 112, float(x1),
                                                          float(y1), scale_x, scale_y)

        # cv2.imwrite(str(counter) + ".jpg", output_pre, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        mae_yaw = abs(yaw_tpre - yaw_truth)
        mae_pitch = abs(pitch_tpre - pitch_truth)
        mae_roll = abs(roll_tpre - roll_truth)
        "================================="
        return output_pre
