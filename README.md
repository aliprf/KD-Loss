	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/facial-landmark-points-detection-using/face-alignment-on-cofw)](https://paperswithcode.com/sota/face-alignment-on-cofw?p=facial-landmark-points-detection-using)

#
Facial Landmark Points Detection Using Knowledge Distillation-Based Neural Networks

## https://aliprf.github.io/KD-Loss/

#### Link to the paper:
Google Scholar:
https://scholar.google.com/citations?view_op=view_citation&hl=en&user=96lS6HIAAAAJ&citation_for_view=96lS6HIAAAAJ:zYLM7Y9cAGgC

Elsevier:
https://www.sciencedirect.com/science/article/pii/S1077314221001582

Arxiv:
https://arxiv.org/abs/2111.07047

#### Link to the paperswithcode.com:
https://paperswithcode.com/paper/facial-landmark-points-detection-using

```diff
@@plaese STAR the repo if you like it.@@
```

```
Please cite this work as:

     @article{fard2022facial,
  title={Facial landmark points detection using knowledge distillation-based neural networks},
  author={Fard, Ali Pourramezan and Mahoor, Mohammad H},
  journal={Computer Vision and Image Understanding},
  volume={215},
  pages={103316},
  year={2022},
  publisher={Elsevier}
}

```

## Introduction
Facial landmark detection is a vital step for numerous facial image analysis applications. Although some deep learning-based methods have achieved good performances in this task, they are often not suitable for running on mobile devices. Such methods rely on networks with many parameters, which makes the training and inference time-consuming. Training lightweight neural networks such as MobileNets are often challenging, and the models might have low accuracy. Inspired by knowledge distillation (KD), this paper presents a novel loss function to train a lightweight Student network (e.g., MobileNetV2) for facial landmark detection. We use two Teacher networks, a Tolerant-Teacher and a Tough-Teacher in conjunction with the Student network. The Tolerant-Teacher is trained using Soft-landmarks created by active shape models, while the tough teacher is trained using the ground truth (aka Hard-landmarks) landmark points. To utilize the facial landmark points predicted by the Teacher networks, we define an Assistive Loss (ALoss) for each Teacher network. Moreover, we define a loss function called KD-Loss that utilizes the facial landmark points predicted by the two pre-trained Teacher networks (EfficientNet-b3) to guide the lightweight Student network towards predicting the Hard-landmarks. Our experimental results on three challenging facial datasets show that the proposed architecture will result in a better-trained Student network that can extract facial landmark points with high accuracy.


##Architecture

We train the Tough-Teacher, and the Tolerant-Teacher networks independently using the Hard-landmarks and the Soft-landmarks respectively utilizing the L2 loss:

![teacher_arch](https://github.com/aliprf/KD-Loss/blob/master/samples/teacher_arch-1.jpg?raw=true)


Proposed KD-based architecture for training the Student network. KDLoss uses the knowledge of the previously trained Teacher networks by utilizing the assistive loss functions ALossT ou and ALossT ol, to improve the performance the face alignment task:

![general_framework](https://github.com/aliprf/KD-Loss/blob/master/samples/general_framework-1.jpg?raw=true)


## Evaluation

Following are some samples in order to show the visual performance of KD-Loss on 300W, COFW and WFLW datasets:

300W:
![KD_300W_samples](https://github.com/aliprf/KD-Loss/blob/master/samples/KD_300W_samples-1.jpg?raw=true)

COFW:
![KD_cofw_samples](https://github.com/aliprf/KD-Loss/blob/master/samples/KD_cofw_samples-1.jpg?raw=true)

WFLW:
![KD_WFLW_samples](https://github.com/aliprf/KD-Loss/blob/master/samples/KD_WFLW_samples-1.jpg?raw=true)

----------------------------------------------------------------------------------------------------------------------------------
## Installing the requirements
In order to run the code you need to install python >= 3.5. 
The requirements and the libraries needed to run the code can be installed using the following command:

```
  pip install -r requirements.txt
```


## Using the pre-trained models
You can test and use the preetrained models using the following codes which are available in the test.py:
The pretrained student model are also located in "models/students".
  
```
  cnn = CNNModel()
        model = cnn.get_model(arch=arch, input_tensor=None, output_len=self.output_len)

        model.load_weights(weight_fname)

        img = None # load a cropped image

        image_utility = ImageUtility()
        pose_predicted = []
        image = np.expand_dims(img, axis=0)

        pose_predicted = model.predict(image)[1][0]
```


## Training Network from scratch


### Preparing Data
Data needs to be normalized and saved in npy format. 

### Training 

### Training Teacher Networks: 

The training implementation is located in teacher_trainer.py class. You can use the following code to start the training for the teacher networks:

```
  '''train Teacher Networks'''
    trainer = TeacherTrainer(dataset_name=DatasetName.w300)
    trainer.train(arch='efficientNet',weight_path=None)
```

### Training Student Networks: 
After Training the teacher networks, you can use the trained teachers to train the student network. The implemetation of training of the student network is provided in teacher_trainer.py . You can use the following code to start the training for the student networks:

```
 st_trainer = StudentTrainer(dataset_name=DatasetName.w300, use_augmneted=True)
    st_trainer.train(arch_student='mobileNetV2', weight_path_student=None,
                     loss_weight_student=2.0,
                     arch_tough_teacher='efficientNet', weight_path_tough_teacher='./models/teachers/ds_300w_ef_tou.h5',
                     loss_weight_tough_teacher=1,
                     arch_tol_teacher='efficientNet', weight_path_tol_teacher='./models/teachers/ds_300w_ef_tol.h5',
                     loss_weight_tol_teacher=1)
```



```
Please cite this work as:

     @article{fard2022facial,
  title={Facial landmark points detection using knowledge distillation-based neural networks},
  author={Fard, Ali Pourramezan and Mahoor, Mohammad H},
  journal={Computer Vision and Image Understanding},
  volume={215},
  pages={103316},
  year={2022},
  publisher={Elsevier}
}

```  

```diff
@@plaese STAR the repo if you like it.@@
```
