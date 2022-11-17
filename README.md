# COVIDNet-CT-Identification

## Dataset 

A large-scale chest CT dataset for COVID-19 detection, comprising 425,024 CT slices from 5,312 patients and 431,205 CT slices from 6,068 patients, respectively.

[Kaggle Link](https://www.kaggle.com/datasets/hgunraj/covidxct)

##### Directory Sturcture

    
    ├── images                     # folder consisting CT scan images
        ├── CP_0_3136_0207.png         
        └── CP_1070_3112_0032.png
        └── ...
    ├── test_COVIDx-CT.txt         # test split
    ├── train_COVIDx-CT.txt        # train split
    ├── val_COVIDx-CT.txt          # validation split



Train set, validation set and  test set are pre-split by ".txt"  label files. Each line in the label files has the following format:

    filename class xmin ymin xmax ymax


Classes are Normal=0, Pneumonia=1, and COVID-19=2. 

Bounding boxes are given in original image coordinates, although the scope of this project does not included predicting the bounding box coordinates.

## Model Training and Testing

The notebook `covidct-2a.ipynb` walks through the process of data preparation and Deep Neural Network model training and testing.

Various backbone architectures can be selected ('vgg16', 'vgg19', 'resnet101', 'resnet152', 'densenet161', 'densenet201') and experimented with different number of layers to be frozen upon fine-tuning. 

The best performance was recieved on densenet 201, using pre-trained weights of ImageNet and fine-tuned by unfreezing last 3 layers.

Data augmentations are also applied during training, for better generalization of Deep Neural Network.

After 3 epochs, following evaluation metrics were achieved :

![image](https://user-images.githubusercontent.com/30556653/172347869-81eac92e-d78f-4f17-9d70-b9c24fc68595.png)

## Model Serving : To do


### References
- [Transfer Learning for Medical Images](https://learnopencv.com/transfer-learning-for-medical-images/)

- [Measuring and tuning performance of a TensorFlow inference system using Triton Inference Server and Tesla T4](https://cloud.google.com/architecture/scalable-tensorflow-inference-system-using-tensorrt-and-tesla-t4)

