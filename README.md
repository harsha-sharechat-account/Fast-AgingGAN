# AgingGAN

This is a face aging deep learning model. Given an input image and a desired age range, the model ages the face to the desired age group. The age groups are:
1. 10-19 (encoded as the integer 0).
2. 20-29 (encoded as the integer 1).
3. 30-39 (encoded as the integer 2).
4. 40-49 (encoded as the integer 3).
5. 50+ (encoded as the integer 4).

It's mostly inspired by the [Identity Preserved Face Aging](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Face_Aging_With_CVPR_2018_paper.pdf). Most changes in this repo are geared towards making the model fast enough for running on a mobile device. It can achieve 30fps on an iPhone X.

# Design
The following image shows the adversarial training setup. The orange arrows denote the path where backprop to the generator happens.

<p align="center">
  
![Fast-AgingGAN](https://user-images.githubusercontent.com/4294680/71646087-5fd13a80-2ce2-11ea-8d5b-055d202ad1f1.png)

</p>

# Requirements
To install the required packages, use the requirements text file like so:
```bash
pip install -r requirements.txt
```

# Dataset
The code is setup to train on the CACD2000 dataset. You would need to download the dataset, and run preprocessing to align and crop the faces from the images. You can use the scripts here to preprocess the dataset: [Scripts](https://github.com/guyuchao/IPCGANs-Pytorch/tree/master/preprocess)

# Pre-trained Model
To try out the pre-trained model as a demo, use the provided pre-trained generator like so:
```bash
python demo.py --image_dir=/path/to/image/directory
```
It will output a matplotlib figure with randomly sampled images from the folder all aged to the 5 age groups defined above.

# Training
To train your own model, run the script like so, after the data has been preprocessed, or your data is in the required format (the dataset text files).
```bash
python main.py --image_dir=/path/to/image/directory --text_dir=/path/where/the/text/files/are --batch_size=24 --epochs=100 --img_size=128 --num_classes=5 --lr=1e-4 --save_iter=200
```
Model checkpoints and training summaries are saved in the directory. You can startup tensorboard to monitor training progress by pointing it to the 'logs' directory that will created.

If you do not have a pretrained age classifier in the models director called 'age_classifier.h5', pre-training will automatically run to train a classifier which will be used for training the generator later.

# Samples
The pretrained model was trained with a small classifier loss, you can increase the aging effect by weighing the loss higher during training. Zoom in to the faces to see the subtle changes. One thing that should be noticed is how well the background is preserved, and also the face identity is preserved compared to traditional methods.

![Sample-1](https://user-images.githubusercontent.com/4294680/71646298-65308400-2ce6-11ea-9234-2c0e738c3b93.png)
![Sample-2](https://user-images.githubusercontent.com/4294680/71646319-de2fdb80-2ce6-11ea-97fa-cc63f58bbb5e.png)

# Contributing
If you have ideas on improving model performance, adding metrics, or any other changes, please make a pull request or open an issue. I'd be happy to accept any contributions.
