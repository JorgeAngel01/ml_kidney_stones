# ml_kidney_stones

- image_manager.ipynb.- This notebook downloads the images as a zip file and rearranges them in a "train" and "test" folder as expected by pytorch's dataloader. The imageManager class receives as parameter the local path where the images are going to be stored, the percentage of images that will be used as the "test" dataset and the "merge" flag. If this flag is set to TRUE it will merge surface and section classes.
- patches_colab_base.ipynb.- Notebook to test AlexNet on the kidney stones (patches) dataset.
- helpers/KidneyImagesLoader.py.- Class to pull the zip file from google drive and returns the data loaders for the training, validation and tests sets. It receives as input the path of the zip file, the percentage of the training set that is going to be used for validation and the transformations to be applied to the train/val sets and the test set.
- helpers/PlotHelper.py.- Class with methods to plot the results given by a CNN model.
- patches_resnet.ipynb.- Notebook to test Resnet34 and ResNet50 on the kidney stones (patches) dataset.
- patches_vgg16.ipynb.- Notebook to test Vgg16 on the kidney stones (patches) dataset.
- patches_googlenet.ipynb.- Notebook to test Google Inception V3 on the kidney stones (patches) dataset.
- statistical_measurements_patches.ipynb.- Notebook to compute mean and standard deviation of rgb channels for each image (patch).
- make-patch.ipynb.- Notebook for automatic patch generation.
- extract-features.ipynb.- Notebook for extract features (eH, eS, eV,LBP) in patches. 

# Python script instructions

There is a python script (run.py) which can be used to train a model, and is meant to be run on the DGX server. 
It is recommended to be executed on a docker container, which can be created with the following command:

```
nvidia-docker run --name [container-name] -it -v [repository-path]:/workspace/kidneystones nvcr.io/nvidia/pytorch:20.11-py3
```

The following packages are required:

- Pythorch-lightning 1.0.2

To download the datasets for the classification of 6 kidney stone classes, open the datasets folder and execute the following command:

```
sh download_6classes.sh
```

The config.yaml file contains the configuration to run a model, the model to use and some parameters can be changed there.

To execute the script we run the following command line, adding the corresponding gpu id.

```
python run.py --gpu 4
```

# Additional Notes
The summary of the experiments can be seen in the following colab page:
https://colab.research.google.com/drive/1Bh8kf1j8oR0bU30hgTowQzoOPnPmYKE0?usp=sharing
