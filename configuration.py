class DatasetName:
    w300 = 'w300'
    cofw = 'cofw'
    wflw = 'wflw'

class DatasetType:
    data_type_train = 0
    data_type_validation = 1
    data_type_test = 2

class LearningConfig:
    batch_size = 70
    epochs = 150

class InputDataSize:
    image_input_size = 224

class WflwConf:
    Wflw_prefix_path = './data/wflw/'  # --> local

    '''     augmented version'''
    augmented_train_pose = Wflw_prefix_path + 'training_set/augmented/pose/'
    augmented_train_annotation = Wflw_prefix_path + 'training_set/augmented/annotations/'
    augmented_train_atr = Wflw_prefix_path + 'training_set/augmented/atrs/'
    augmented_train_image = Wflw_prefix_path + 'training_set/augmented/images/'
    augmented_train_tf_path = Wflw_prefix_path + 'training_set/augmented/tf/'
    '''     original version'''
    no_aug_train_annotation = Wflw_prefix_path + 'training_set/no_aug/annotations/'
    no_aug_train_atr = Wflw_prefix_path + 'training_set/no_aug/atrs/'
    no_aug_train_pose = Wflw_prefix_path + 'training_set/no_aug/pose/'
    no_aug_train_image = Wflw_prefix_path + 'training_set/no_aug/images/'
    no_aug_train_tf_path = Wflw_prefix_path + 'training_set/no_aug/tf/'

    orig_number_of_training = 7500
    orig_number_of_test = 2500

    augmentation_factor = 4  # create . image from 1
    augmentation_factor_rotate = 15  # create . image from 1
    num_of_landmarks = 98

class CofwConf:
    Cofw_prefix_path = './data/cofw/'  # --> local

    augmented_train_pose = Cofw_prefix_path + 'training_set/augmented/pose/'
    augmented_train_annotation = Cofw_prefix_path + 'training_set/augmented/annotations/'
    augmented_train_image = Cofw_prefix_path + 'training_set/augmented/images/'
    augmented_train_tf_path = Cofw_prefix_path + 'training_set/augmented/tf/'

    orig_number_of_training = 1345
    orig_number_of_test = 507

    augmentation_factor = 10
    num_of_landmarks = 29


class D300wConf:
    w300w_prefix_path = './data/300W/'  # --> local

    orig_300W_train = w300w_prefix_path + 'orig_300W_train/'
    augmented_train_pose = w300w_prefix_path + 'training_set/augmented/pose/'
    augmented_train_annotation = w300w_prefix_path + 'training_set/augmented/annotations/'
    augmented_train_image = w300w_prefix_path + 'training_set/augmented/images/'
    augmented_train_tf_path = w300w_prefix_path + 'training_set/augmented/tf/'

    no_aug_train_annotation = w300w_prefix_path + 'training_set/no_aug/annotations/'
    no_aug_train_pose = w300w_prefix_path + 'training_set/no_aug/pose/'
    no_aug_train_image = w300w_prefix_path + 'training_set/no_aug/images/'
    no_aug_train_tf_path = w300w_prefix_path + 'training_set/no_aug/tf/'

    orig_number_of_training = 3148
    orig_number_of_test_full = 689
    orig_number_of_test_common = 554
    orig_number_of_test_challenging = 135

    augmentation_factor = 4  # create . image from 1
    num_of_landmarks = 68

