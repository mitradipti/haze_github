import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

class EncoderBlock(nn.Module):
    def __init__(self, in_channels):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,64, kernel_size=3,padding=1)
        self.conv2a = nn.Conv2d(64,64, kernel_size=1,padding=0)
        self.conv2b = nn.Conv2d(64,64, kernel_size=3,padding=1) #gave kernel size 3x3 instead of 1x1 to match the dimension with keras
        self.conv3 = nn.Conv2d(128,128, kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(128,128, kernel_size=3,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.leaky_relu = nn.LeakyReLU(negative_slope = 0.1)

    def forward(self,x):
# First layer in the forward path
        x=self.conv1(x)
        x=self.leaky_relu(x)
# Inception Module (Two parallel convolutional layers)
        conv2a=self.conv2a(x)
        conv2a=self.leaky_relu(conv2a)
        conv2b=self.conv2b(x)
        conv2b=self.leaky_relu(conv2b)
        inception_output=torch.cat([conv2a,conv2b],dim=1)
# Dense connections within the block
        x=self.conv3(inception_output)
        x=self.leaky_relu(x)
        x=self.conv4(x)
        x=self.leaky_relu(x)
# Pooling layer for downsampling
        pool=self.pool(x)
# Skip connections to the decoder block
        skip_connection = pool
        return skip_connection,pool

class BottleneckBlock(nn.Module):
    def __init__(self):
        super(BottleneckBlock, self).__init__()
# Three convolutional layers in the forward path
        self.conv1 = nn.Conv2d(128,256,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(256,256,kernel_size=3,padding=1)
# Inverse convolutional layer for up-sampling
        self.upsample = nn.ConvTranspose2d(256,128,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, input_tensor, skip_connection):
        conv1 = self.leaky_relu(self.conv1(input_tensor))
        conv2 = self.leaky_relu(self.conv2(conv1))
        conv3 = self.leaky_relu(self.conv3(conv2))
        upsample= self.leaky_relu(self.upsample(conv3))
        upsample=F.interpolate(upsample,size=(skip_connection.shape[2],skip_connection.shape[3]))
# concatenate with the feature map from the skip connection
        concatenated=torch.cat((upsample, skip_connection), dim=1)
        return concatenated

class DecoderBlock(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(DecoderBlock,self).__init__()
# inverse conv layer for up-sampling
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv1 = nn.Conv2d(out_channels+2*128,256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256+2*128,256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256+2*128,256, kernel_size=3, padding=1)
        self.leaky_relu= nn.LeakyReLU(negative_slope=0.1)

    def forward(self, input_tensor, skip_connections):
        upsample=self.leaky_relu(self.upsample(input_tensor))
        upsampled_skip_connections=[F.interpolate(skip, size=(upsample.shape[2], upsample.shape[3])) if skip.shape[2]!= upsample.shape[2] else skip for skip in skip_connections]
# concatenate all skip connections with the up-sampled feature map
        concatenated=torch.cat([upsample] + upsampled_skip_connections, dim=1)

        conv1 = self.leaky_relu(self.conv1(concatenated))
# concatenate skip connection with the input of the second and third layer
        conv2_input= torch.cat([conv1] + upsampled_skip_connections, dim=1)
        conv2 = self.leaky_relu(self.conv2(conv2_input))

        conv3_input= torch.cat([conv2] + upsampled_skip_connections, dim=1)
        conv3 = self.leaky_relu(self.conv3(conv3_input))
        return conv3

class FullyConnectedBlock(nn.Module):
    def __init__(self, in_channels, aux_channels):
        super(FullyConnectedBlock,self).__init__()
# conv1 layer with residual connection to auxiliary neural network
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3,padding=1)
        self.conv_aux = nn.Conv2d(aux_channels, 256, kernel_size=3,padding=1, stride=2)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3,padding=1)
        self.conv3= nn.Conv2d(256, 256, kernel_size=3,padding=1)
        self.final_conv = nn.Conv2d(256, 1, kernel_size=1)
        self.leaky_relu= nn.LeakyReLU(negative_slope=0.1)

    def forward(self, input_tensor, auxiliary_input):
        conv1=self.leaky_relu(self.conv1(input_tensor))

        auxiliary_input_resized=self.leaky_relu(self.conv_aux(auxiliary_input))

        if conv1[2:] != auxiliary_input_resized.shape[2:]:
          auxiliary_input_resized=F.interpolate(auxiliary_input_resized,size=conv1.shape[2:])
# residual connection to auxiliary neural network
        residual_connection=torch.cat([conv1, auxiliary_input_resized], dim=1)
        conv2 = self.leaky_relu(self.conv2(residual_connection))
        conv3= self.leaky_relu(self.conv3(conv2))
# Final layer to produce 1-channel output
        output = self.final_conv(conv3)
        return output
