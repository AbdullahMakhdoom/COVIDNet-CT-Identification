# COVIDNet-CT-Identification

## Dataset 

A large-scale chest CT dataset for COVID-19 detection, comprising of 104,009 CT slices from 1,489 patients. 

[Kaggle Link](https://www.kaggle.com/datasets/c395fb339f210700ba392d81bf200f766418238c2734e5237b5dd0b6fc724fcb/version/1)

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


Classes are Normal=0, Pneumonia=1, and COVID-19=2. Bounding boxes are given in original image coordinates.

## Model Training and Testing

The notebook `covidct-2a.ipynb` walks through the process of data preparation and Deep Neural Network model training and testing.

Various backbone architectures can be selected ('vgg16', 'vgg19', 'resnet101', 'resnet152', 'densenet161', 'densenet201') and experimented with different number of layers to be frozen upon fine-tuning. 

Data augmentations are also applied during training, for better generalization of DNN.

After 3 epochs, following evaluation metrics were achieved :

![image](https://user-images.githubusercontent.com/30556653/172347869-81eac92e-d78f-4f17-9d70-b9c24fc68595.png)

## Model Serving


### References
- [Transfer Learning for Medical Images](https://learnopencv.com/transfer-learning-for-medical-images/)

- [Measuring and tuning performance of a TensorFlow inference system using Triton Inference Server and Tesla T4](https://cloud.google.com/architecture/scalable-tensorflow-inference-system-using-tensorrt-and-tesla-t4)









 
