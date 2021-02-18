
from configuration import DatasetName, D300wConf, InputDataSize, CofwConf, WflwConf
from cnn_model import CNNModel
from pca_utility import PCAUtility
from image_utility import ImageUtility
from student_train import StudentTrainer
from test import Test
from train import Train

if __name__ == '__main__':

    '''test models'''

    '''train Teacher Networks'''
    trainer = Train(dataset_name=DatasetName.w300,
                    arch='efficientNet',
                    weight=None,
                    accuracy=100)

    '''Training Student Network'''
    '''300W'''
    st_trainer = StudentTrainer(dataset_name=DatasetName.w300, use_augmneted=True)
    st_trainer.train(arch_student='mobileNetV2', weight_path_student=None,
                     loss_weight_student=2.0,
                     arch_tough_teacher='efficientNet', weight_path_tough_teacher='./models/teachers/ds_300w_ef_tou.h5',
                     loss_weight_tough_teacher=1,
                     arch_tol_teacher='efficientNet', weight_path_tol_teacher='./models/teachers/ds_300w_ef_tol.h5',
                     loss_weight_tol_teacher=1)

    '''COFW'''
    st_trainer = StudentTrainer(dataset_name=DatasetName.cofw, use_augmneted=True)
    st_trainer.train(arch_student='mobileNetV2', weight_path_student=None,
                     loss_weight_student=2.0,
                     arch_tough_teacher='efficientNet', weight_path_tough_teacher='./models/teachers/ds_cofw_ef_tou.h5',
                     loss_weight_tough_teacher=1,
                     arch_tol_teacher='efficientNet', weight_path_tol_teacher='./models/teachers/ds_cofw_ef_tol.h5',
                     loss_weight_tol_teacher=1)

    '''WFLW'''
    st_trainer = StudentTrainer(dataset_name=DatasetName.wflw, use_augmneted=True)
    st_trainer.train(arch_student='mobileNetV2', weight_path_student=None,
                     loss_weight_student=2.0,
                     arch_tough_teacher='efficientNet', weight_path_tough_teacher='./models/teachers/ds_wflw_ef_tou.h5',
                     loss_weight_tough_teacher=1,
                     arch_tol_teacher='efficientNet', weight_path_tol_teacher='./models/teachers/ds_wflw_ef_tol.h5',
                     loss_weight_tol_teacher=1)