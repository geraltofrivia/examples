This script would train a Pytorch implementation of [DCGAN](https://arxiv.org/abs/1511.06434) on Oxford-102 Flowers dataset.

### Task
 **3** -  Train DCGAN on Flowers

### Usage
Get the dataset from [link](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/)

Unpack it in ``./dataset/flower/jpg/<anyarbitaryDIRname>``

Run 
``
    python dcgan.py --dataset imagefolder --cuda --gpuid 2 --manualSeed 42 --workers 2 --dataroot ./dataset/flower/jpg/
``

> Script runs for 25 epochs by default

> if you want to continue training using some saved models, use the --netG, --netD flags.

> use --niter if you want to train it for more than 25 epochs. (Training is fast, so feel free)

### Outputs
The outputs of this script are saved in `../op/q1/` directory.