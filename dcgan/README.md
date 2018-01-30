This script would train a Pytorch implementation of [DCGAN](https://arxiv.org/abs/1511.06434) on imagenet.

### Task
 **1.1** -  Train DCGAN on Imagenet1k

### Usage

``
    python main.py --dataset imagenet --cuda --gpuid 2 --manualSeed 42 --workers 2 
``

> Path to dataset is pre-specified in the script.

> if you want to continue training using some saved models, use the --netG, --netD flags.

### Outputs
The outputs of this script are saved in `../op/q1/` directory.