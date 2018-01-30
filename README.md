# DC-GAN Project for DLVR Lab

### Tasks:

 1. Train DCGAN on Imagenet1k; Use LinearSVM to evaluate the discriminator on Food101
> One common technique for evaluating the quality of unsupervised representation learning algorithms is to apply them as a feature extractor on supervised datasets and evaluate the performance of linear models trained using these features. Do classification on Food-101 dataset [5] (see Project 1) using GANs trained on the IMAGENET-1K as a feature extractor. Train a linear SVM for classification using these features and report your classification results. You can use the lib-SVM package. 
 
 2. Train DCGAN on any LSUN class. (We chose Towers)
> Train a generator of yourself using another dataset. You are free to choose any one of the datasets listed here except the Bedroom dataset since it is used in the paper.

 3. Train DCGAN; and perform vector arithmetic op over them. (We chose Oxford-102 flowers dataset)
 > Train a generator of yourself using any dataset and or topic you prefer. Generate images using your own trained GAN, and try to get similar vector arithmetics as introduced
in [11] (see ‘vector arithmetic in face samples’ section). Feel free to create your own dataset of interest; other possibilities include Caltech-UCSD Birds-200-2011, Oxford-102
flowers and Food-101.

### Usage:
 Head over to ```./dcgan/``` for Task1; ```./dcgan_lsun/``` for Task2; and ```./vectormagic/``` for the third task.


### Reference:

**[Paper](https://arxiv.org/abs/1511.06434)**

**[Assignment](https://github.com/geraltofrivia/pytorchexamples/tree/master/assignment.pdf)**

**Datasets**: [Imagenet](http://www.image-net.org/); [Food-101](https://www.vision.ee.ethz.ch/datasets_extra/food-101/); [Oxford-102 Flower](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/)