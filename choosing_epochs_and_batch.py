import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns
import logging


def get_logger():
    logger = logging.getLogger('train_script')
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def train_net(n_epochs, network,  print_train = True):

    # prepare the net for training
    network.train()
    results = []

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0

        # train on batches of data, assumes you already have train_loader
        for batch_i, data in enumerate(train_loader):
            # get the input images and their corresponding labels
            images = data['image']
            key_pts = data['keypoints']

            # flatten pts
            key_pts = key_pts.view(key_pts.size(0), -1)

            # cuda flags
            if torch.cuda.is_available():
                run_images = images.type(torch.FloatTensor).to('cuda')
                run_key_pts = key_pts.type(torch.FloatTensor).to('cuda')
            else:
                logger.debug('moving images to cpu for epoch {}'.format(epoch))
                run_images = images.type(torch.FloatTensor)
                run_key_pts = key_pts.type(torch.FloatTensor)

            # forward pass to get outputs
            output_pts = network(run_images)

            # calculate the loss between predicted and target keypoints
            loss = criterion(output_pts, run_key_pts)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()

            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # print loss statistics
            running_loss += loss.item()
            if batch_i % 10 == 9:    # print every 10 batches
                if print_train:
                    logger.debug('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i +1, running_loss /10))
                    results.append([epoch + 1, batch_i +1, running_loss /10])
                running_loss = 0.0

    logger.info('Finished Training')
    return results


def analyse_results(network):
    SmoothL1Loss = []
    L1LossList = []
    MSELossList = []

    logger.debug('moving network to cpu')
    infer_net = network.to('cpu')

    SmoothL1 = nn.SmoothL1Loss()
    L1Loss = nn.L1Loss()
    MSELoss = nn.MSELoss()

    for i, sample in enumerate(test_loader):
        # get sample data: images and ground truth keypoints
        #logger.debug('moving data to cpu')
        images = sample['image'].type(torch.FloatTensor).to('cpu')
        key_pts = sample['keypoints'].type(torch.FloatTensor).to('cpu')

        # forward pass to get net output
        output_pts = infer_net(images)

        # reshape to batch_size x 68 x 2 pts
        output_pts = output_pts.view(output_pts.size()[0], 68, -1)

        SmoothL1num = SmoothL1(output_pts, key_pts)
        SmoothL1Loss.append(SmoothL1num.data.numpy())

        L1Lossnum = L1Loss(output_pts, key_pts)
        L1LossList.append(L1Lossnum.data.numpy())

        MSELossnum = MSELoss(output_pts, key_pts)
        MSELossList.append(MSELossnum.data.numpy())

    return SmoothL1Loss, L1LossList, MSELossList


####################### Main Routine

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import itertools

# the dataset we created in Notebook 1 is copied in the helper file `data_load.py`
from data_load import FacialKeypointsDataset
# the transforms we defined in Notebook 1 are in the helper file `data_load.py`
from data_load import Rescale, RandomCrop, Normalize, ToTensor

logger = get_logger()

####################### Load Data ###################

data_transform = transforms.Compose([Rescale(250),
                                     RandomCrop(224),
                                     Normalize(),
                                     ToTensor()])

# testing that you've defined a transform
assert(data_transform is not None), 'Define a data_transform'

######################## Transform Data #############

# create the transformed dataset
transformed_dataset = FacialKeypointsDataset(csv_file='data/training_frames_keypoints.csv',
                                             root_dir='data/training/',
                                             transform=data_transform)

# Parameter
n_epochs = 1

##### test param
#####
#####
batch_sizes = [10, 20, 30, 40]

from models import Net_drop_more_layer

experiments = []
for batch in batch_sizes:

    # Train Loader
    train_loader = DataLoader(transformed_dataset,
                              batch_size=batch,
                              shuffle=True,
                              num_workers=0)

    # Test Loader
    # create the test dataset
    test_dataset = FacialKeypointsDataset(csv_file='data/test_frames_keypoints.csv',
                                                 root_dir='data/test/',
                                                 transform=data_transform)

    test_loader = DataLoader(test_dataset,
                              batch_size=batch,
                              shuffle=True,
                              num_workers=0)

######################### Train Network ###############

    ########################## Make sure we have a fresh net each iteration
    net = Net_drop_more_layer()

    if torch.cuda.is_available():
        logger.debug('CUDA Training')
        net.to('cuda')

    logger.info('running with batch: {}'.format(batch))
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, amsgrad=True)

    train_net(n_epochs, net, print_train = False)
    SmoothL1Loss, L1LossList, MSELossList = analyse_results(net)

    print('batch size is: {}'.format(batch))
    sns.distplot(SmoothL1Loss, label = 'batch: {}'.format(batch))

    print('Smooth L1 {}'.format(np.mean(SmoothL1Loss)))
    print('L1 {}'.format(np.mean(L1LossList)))
    print('SME L1 {}'.format(np.mean(MSELossList)))

    experiments.append({'name': 'batch: {}'.format(batch),
                        'Smooth-L1': SmoothL1Loss,
                        'L1': L1LossList,
                        'SME': MSELossList})

    del net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


############################# Epoch test

batch_size = 30

# Train Loader
train_loader = DataLoader(transformed_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)

# Test Loader
# create the test dataset
test_dataset = FacialKeypointsDataset(csv_file='data/test_frames_keypoints.csv',
                                                 root_dir='data/test/',
                                                 transform=data_transform)

test_loader = DataLoader(test_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)


##### test param
#####
#####
n_epoch_test = [10, 20. 40, 80]


for epoch_test in n_epoch_test:

######################### Train Network ###############

    ########################## Make sure we have a fresh net each iteration
    net = Net_drop_more_layer()

    if torch.cuda.is_available():
        logger.debug('CUDA Training')
        net.to('cuda')

    logger.info('running with epoch: {}'.format(epoch_test))
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, amsgrad=True)

    train_net(epoch_test, net, print_train = False)
    SmoothL1Loss, L1LossList, MSELossList = analyse_results(net)

    print('epochs are: {}'.format(epoch_test))
    sns.distplot(SmoothL1Loss, label = 'epochs: {}'.format(epoch_test))

    print('Smooth L1 {}'.format(np.mean(SmoothL1Loss)))
    print('L1 {}'.format(np.mean(L1LossList)))
    print('SME L1 {}'.format(np.mean(MSELossList)))

    experiments.append({'name': 'epochs: {}'.format(epoch_test),
                        'Smooth-L1': SmoothL1Loss,
                        'L1': L1LossList,
                        'SME': MSELossList})

    del net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()