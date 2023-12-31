import torch.nn as nn
import torch.nn.functional as F
import torch
from activations import *
from torchvision import models


def get_resnet18(num_classes):
    model = models.resnet18(pretrained=True)
    n_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, num_classes))#, nn.LogSoftmax(dim=1))
    #instead normalization prepend batchnorm
    model = nn.Sequential(nn.BatchNorm2d(num_features=3, affine=False), model)
    return model

    
# https://www.kaggle.com/code/shadabhussain/cifar-10-cnn-using-pytorch
class CIFAR_target_net(nn.Module):
    def __init__(self):
        super(CIFAR_target_net, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 4 x 4

            nn.Flatten(), 
            nn.Linear(256*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10))

    def forward(self, x):
        return self.network(x)



# Target Model definition
class MNIST_target_net(nn.Module):
    def __init__(self):
        super(MNIST_target_net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)

        self.fc1 = nn.Linear(64*4*4, 200)
        self.fc2 = nn.Linear(200, 200)
        self.logits = nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64*4*4)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc2(x))
        x = self.logits(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, image_nc):
        super(Discriminator, self).__init__()
        model = [
            #c8 3x32x32
            nn.Conv2d(image_nc, 8, kernel_size=4, stride=2, padding=0, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # AReLU(),
            # Rational(),
            # Swish(),
            # c16 8x15x15
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            # AReLU(),
            # Rational(),
            # Swish(),
            # c32 16x6x6
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # AReLU(),
            # Rational(),     
            # Swish(), 
            # 32x2x2      

            nn.Conv2d(32, 1, 1, bias=True),
            # 1x2x2
        ]
        self.model = nn.Sequential(*model)
        self.prob = nn.Sigmoid()

    def forward(self, x):
        output = self.model(x).squeeze() 
        probs = self.prob(output)
        return output, probs

class DiscriminatorMNIST(nn.Module):
    def __init__(self, image_nc):
        super(DiscriminatorMNIST, self).__init__()
        model = [
            #c8
            nn.Conv2d(image_nc, 8, kernel_size=3, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # AReLU(),
            # Rational(),
            # Swish(),
            # c16
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            # AReLU(),
            # Rational(),
            # Swish(),
            # c32
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # AReLU(),
            # Rational(),     
            # Swish(),       

            nn.Conv2d(32, 1, 1, bias=True),
        ]
        self.model = nn.Sequential(*model)
        self.prob = nn.Sigmoid()

    def forward(self, x):
        output = self.model(x).squeeze() 
        probs = self.prob(output)
        return output, probs

class Generator(nn.Module):
    def __init__(self,
                 gen_input_nc,
                 image_nc,
                 ):
        super(Generator, self).__init__()
        #encoder gom 3 conv2d theo sau la normalize, relu
        encoder_lis = [
            # MNIST:1*28*28 CIFAR 3*32*32
            nn.Conv2d(gen_input_nc, 8, kernel_size=3, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            # nn.LeakyReLU(0.2, inplace=True),
            # AReLU(),
            # Rational(),
            # Swish(),
            # 8*26*26 cifar 8*30*30
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            # nn.LeakyReLU(0.2, inplace=True),
            # AReLU(),
            # Rational(),
            # Swish(),
            # 16*12*12 cf 16*14*14
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            # nn.LeakyReLU(0.2, inplace=True),
            # AReLU(),
            # Rational(),
            # Swish(),
            # 32*5*5 cf 32*6*6
        ]
        #bottleneck gom 4 resnetblock
        bottle_neck_lis = [ResnetBlock(32),
                       ResnetBlock(32),
                       ResnetBlock(32),
                       ResnetBlock(32),]
        #decode gom 3 convtranspose2d theo sau la normalize, relu
        decoder_lis = [
            #32*6*6
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            # nn.LeakyReLU(0.2, inplace=True),
            # AReLU(),
            # Rational(),
            # Swish(),

            # state size. 16 x 11 x 11 # 16 x 13 x 13
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            # nn.LeakyReLU(0.2, inplace=True),
            # AReLU(),
            # Rational(),
            # Swish(),

            # state size. 8 x 23 x 23 # 8x27x27
            nn.ConvTranspose2d(8, image_nc, kernel_size=6, stride=1, padding=0, bias=False),
            nn.Tanh()
            # state size. image_nc x 28 x 28 # 3x32x32
        ]

        self.encoder = nn.Sequential(*encoder_lis)
        self.bottle_neck = nn.Sequential(*bottle_neck_lis)
        self.decoder = nn.Sequential(*decoder_lis)

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottle_neck(x)
        x = self.decoder(x)
        return x 

class GeneratorNoTrans(nn.Module):
    def __init__(self,
                 gen_input_nc,
                 image_nc,
                 ):
        super(GeneratorNoTrans, self).__init__()

        encoder_lis = [
            # MNIST:1*28*28
            nn.Conv2d(gen_input_nc, 8, kernel_size=3, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            # nn.LeakyReLU(0.2, inplace=True),
            # AReLU(),
            # Rational(),
            # Swish(),
            # 8*26*26 cf 8*30*30
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            # nn.LeakyReLU(0.2, inplace=True),
            # AReLU(),
            # Rational(),
            # Swish(),
            # 16*12*12 cf 16*14*14
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            # nn.LeakyReLU(0.2, inplace=True),
            # AReLU(),
            # Rational(),
            # Swish(),
            # 32*5*5 cf 32*6*6
        ]

        bottle_neck_lis = [ResnetBlock(32),
                       ResnetBlock(32),
                       ResnetBlock(32),
                       ResnetBlock(32),]

        decoder_lis = [
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=0, bias=False),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            # nn.LeakyReLU(0.2, inplace=True),
            # AReLU(),
            # Rational(),
            # Swish(),

            # state size. 16 x 11 x 11
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            # nn.LeakyReLU(0.2, inplace=True),
            # AReLU(),
            # Rational(),
            # Swish(),

            # state size. 8 x 23 x 23
            nn.Conv2d(8, image_nc, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
            # state size. image_nc x 28 x 28
        ]

        self.encoder = nn.Sequential(*encoder_lis)
        self.bottle_neck = nn.Sequential(*bottle_neck_lis)
        self.decoder = nn.Sequential(*decoder_lis)

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottle_neck(x)
        x = self.decoder(x)
        return x 
    

# Define a resnet block
# modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out