import os
import numpy as np

# Importing torch stuff
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.autograd import Variable


# Macros (Based on Flower dataset  params)
nz = 100
ngf = 64
nc = 3

# Paths for pre-trained Generator
netG_path = "../op/q3/run1/netG_epoch_24.pth"
op_path = "../op/q3/run1/quickmath/"

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class _netG(nn.Module):
    def __init__(self):
        super(_netG, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        output = self.main(input)
        return output

# Load the discriminator using the path (defined above)
netG = _netG()
netG.apply(weights_init)
netG.load_state_dict(torch.load(netG_path, map_location=lambda storage, loc: storage))
print(netG)


def generate_with_noise(z):
    """Expects a numpy noise arr (with shape: 1,100,1,1)"""

    # Convert the variable to a torch tensor, then a torch variable
    noise = Variable(torch.from_numpy(z).float())

    # Then give it to generator to get hot steamy image
    image = netG(noise)

    return image


def deal_with_user():
    """
        The main script with interactive loops

    :return:
    """

    selected_a = False
    selected_b = False
    selected_c = False

    images_a, images_b, images_c = [], [], []

    while True:

        if selected_a and selected_b and selected_c:
            # The user has selected all the three images, now can break the interactive loop
            break

        if not selected_a:

            # Create three noise variables, generate three corresponding images and ask user to verify them.
            z = [np.random.normal(0, 1, size=(1, 100, 1, 1)) for _ in range(3)]

            # Create the three corresponding images
            images_a = [generate_with_noise(_noise) for _noise in z]

            # Dump them in the op dir
            vutils.save_image(images_a[0].data, op_path + 'a_00.png', normalize=True)
            vutils.save_image(images_a[1].data, op_path + 'a_01.png', normalize=True)
            vutils.save_image(images_a[2].data, op_path + 'a_02.png', normalize=True)

            print("Generated images for A.")

        if not selected_b:

            # Create three noise variables, generate three corresponding images and ask user to verify them.
            z = [np.random.normal(0, 1, size=(1, 100, 1, 1)) for _ in range(3)]

            # Create the three corresponding images
            images_b = [generate_with_noise(_noise) for _noise in z]

            # Dump them in the op dir
            vutils.save_image(images_b[0].data, op_path + 'b_00.png', normalize=True)
            vutils.save_image(images_b[1].data, op_path + 'b_01.png', normalize=True)
            vutils.save_image(images_b[2].data, op_path + 'b_02.png', normalize=True)

            print("Generated images for B.")

        if not selected_c:

            # Create three noise variables, generate three corresponding images and ask user to verify them.
            z = [np.random.normal(0, 1, size=(1, 100, 1, 1)) for _ in range(3)]

            # Create the three corresponding images
            images_c = [generate_with_noise(_noise) for _noise in z]

            # Dump them in the op dir
            vutils.save_image(images_c[0].data, op_path + 'c_00.png', normalize=True)
            vutils.save_image(images_c[1].data, op_path + 'c_01.png', normalize=True)
            vutils.save_image(images_c[2].data, op_path + 'c_02.png', normalize=True)

            print("Generated images for C.")

        # Check the remaining options left
        remaining = ''
        if not selected_a: remaining += 'a'
        if not selected_b: remaining += 'b'
        if not selected_c: remaining += 'c'

        # Prompts and Inputs
        print("Generated new batch of images. Do check")
        print("Amongst %s, select the ones that you like. " % remaining)
        print("To do so, type the associated char in the prompt below")
        input = raw_input("Prompt:\t")

        # Parse the input
        if 'a' in input: selected_a = True
        if 'b' in input: selected_b = True
        if 'c' in input: selected_c = True

    return images_a, images_b, images_c


if __name__ == '__main__':
    deal_with_user()