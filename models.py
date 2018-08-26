## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):
    # Based on https://arxiv.org/pdf/1511.04031.pdf

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 5, stride=1, padding=2)  # add padding to keep image size
        self.batch1 = nn.BatchNorm2d(16, 0.9)
        self.conv2 = nn.Conv2d(16, 48, 3, stride=1, padding=1)

        self.mpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(48, 64, 3)

        self.conv4 = nn.Conv2d(64, 64, 2)

        self.relu = nn.ReLU(inplace=True)
        # self.batchnorm = nn.BatchNorm2d(32)

        # self.fc_lin1 = nn.Linear(640000, 1024)

        self.fc_lin2 = nn.Linear(43264, 136)

    def forward(self, x):

        ## x = self.pool(F.relu(self.conv1(x)))
        # x = self.relu(self.conv1(x))
        x = self.relu(self.batch1(self.conv1(x)))
        x = self.mpool(x)
        x = self.mpool(self.relu(self.conv2(x)))

        x = self.mpool(self.relu(self.conv3(x)))
        x = self.relu(self.conv4(x))

        x = x.view(x.size(0), -1)
        x = F.dropout(x, training=self.training)
        # x = self.fc_lin1(x)
        x = self.fc_lin2(x)

        return x


class Net_wo_drop(nn.Module):
    # Version of Net with no dropout layers

    def __init__(self):
        super(Net_wo_drop, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 5, stride=1, padding=2)  # add padding to keep image size
        self.conv2 = nn.Conv2d(16, 48, 3, stride=1, padding=1)
        self.mpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(48, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 2)

        self.relu = nn.ReLU(inplace=True)
        self.fc_lin2 = nn.Linear(43264, 136)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.mpool(x)
        x = self.mpool(self.relu(self.conv2(x)))

        x = self.mpool(self.relu(self.conv3(x)))
        x = self.relu(self.conv4(x))

        x = x.view(x.size(0), -1)
        x = self.fc_lin2(x)

        return x


class Net_drop(nn.Module):
    # version of Net with normalisation

    def __init__(self):
        super(Net_drop, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 5, stride=1, padding=2)  # add padding to keep image size
        self.batch1 = nn.BatchNorm2d(16, 0.9)
        self.conv2 = nn.Conv2d(16, 48, 3, stride=1, padding=1)
        self.batch2 = nn.BatchNorm2d(48, 0.9)
        self.mpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(48, 64, 3)

        self.conv4 = nn.Conv2d(64, 64, 2)

        self.relu = nn.ReLU(inplace=True)
        self.fc_lin2 = nn.Linear(43264, 136)

    def forward(self, x):
        x = self.relu(self.batch1(self.conv1(x)))
        x = self.mpool(x)
        x = self.mpool(self.relu(self.batch2(self.conv2(x))))

        x = self.mpool(self.relu(self.conv3(x)))
        x = self.relu(self.conv4(x))

        x = x.view(x.size(0), -1)
        x = F.dropout(x, training=self.training)
        x = self.fc_lin2(x)

        return x


class Net_drop_more_layer(nn.Module):
    # version of net with more layers and normalisation

    def __init__(self):
        super(Net_drop_more_layer, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 5, stride=1, padding=2)  # add padding to keep image size
        self.batch1 = nn.BatchNorm2d(16, 0.9)

        self.conv2 = nn.Conv2d(16, 48, 3, stride=1, padding=1)
        self.batch2 = nn.BatchNorm2d(48, 0.9)

        self.conv3 = nn.Conv2d(48, 64, 3)
        self.batch3 = nn.BatchNorm2d(64, 0.9)

        self.conv4 = nn.Conv2d(64, 96, 2)
        self.batch4 = nn.BatchNorm2d(96, 0.9)

        self.conv5 = nn.Conv2d(96, 96, 2)

        self.mpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.relu = nn.ReLU(inplace=True)
        self.fc_lin2 = nn.Linear(13824, 136)

    def forward(self, x):
        x = self.mpool(self.relu(self.batch1(self.conv1(x))))
        x = self.mpool(self.relu(self.batch2(self.conv2(x))))
        x = self.mpool(self.relu(self.batch3(self.conv3(x))))
        x = self.mpool(self.relu(self.batch4(self.conv4(x))))

        x = self.relu(self.conv5(x))

        x = x.view(x.size(0), -1)
        x = F.dropout(x, training=self.training)
        x = self.fc_lin2(x)
        return x


class NaimNet(nn.Module):
    # based on https://arxiv.org/pdf/1710.00977.pdf

    def __init__(self):
        super(NaimNet, self).__init__()


        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 4, stride=1, padding=0)  # add padding to keep image size
        self.drop1 = nn.Dropout(p=0.1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=0)
        self.drop2 = nn.Dropout(p=0.2)
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.drop3 = nn.Dropout(p=0.3)
        self.conv4 = nn.Conv2d(128, 256, 1)
        self.drop4 = nn.Dropout(p=0.4)

        self.mpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.elu = nn.ELU(inplace=True)

        self.drop5 = nn.Dropout(p=0.5)
        self.drop6 = nn.Dropout(p=0.6)

        self.fc_lin1 = nn.Linear(43264, 1000)
        self.fc_lin2 = nn.Linear(1000, 1000)

        self.fc_lin3 = nn.Linear(1000, 136)

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        # x = self.relu(self.conv1(x))
        x = self.drop1(self.mpool(self.elu(self.conv1(x))))
        x = self.drop2(self.mpool(self.elu(self.conv2(x))))
        x = self.drop3(self.mpool(self.elu(self.conv3(x))))
        x = self.drop4(self.mpool(self.elu(self.conv4(x))))

        x = x.view(x.size(0), -1)

        # x = F.dropout(x,training=self.training)
        x = self.drop5(self.elu(self.fc_lin1(x)))
        x = self.drop6(self.elu(self.fc_lin2(x)))
        x = self.fc_lin3(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
