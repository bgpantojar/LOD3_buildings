import torch.nn as nn
import torch.nn.init as init
from .global_variables import *

device = DEVICE

def xavier(param):
    #Used for initializing weights of extras, loc and conf layes in the ssd
    init.xavier_uniform_(param)


def weights_init(m):
    #Used for initializing weights of extras, loc and conf layes in the ssd
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

#https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py
class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.batch_losses = []
        self.batch_avg_losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(train_loader, model, loss_function, optimizer, epoch):
    """
    Defines the behaviour of training - 1 epoch.
    :train_loader  - DataLoader of the Class TrainDataset
    model          - SSD network
    loss_function: - MultiBox loss
    optimizer:     - optimizer
    epoch:         - epoch number
    """
    #Train Model
    model.train()

    #Create an Average Meter() to keep up the losses and time
    losses = AverageMeter()

    #For each batch of imgs
    for i, (images, boxes, labels) in enumerate(train_loader):

        #To CUDA - GPU training
        images = images.to(device)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        #Get Predictions out of the model - Forward Propagation
        predicted_locs, predicted_scores = model(images)

        #Compute loss using MultiBoxLoss function
        loss = loss_function(predicted_locs, predicted_scores, boxes, labels)


        #Backward Propagation
        optimizer.zero_grad()
        loss.backward()

        #Update model
        optimizer.step()

        #Set Up
        losses.update(loss.item(), images.size(0))
    
    print('Epoch: {0}\t'
          'AVG Loss Train {loss.avg:.3f}'.format(epoch, loss=losses))
    # free up memory
    del predicted_locs, predicted_scores, images, boxes, labels
    
    return losses


def validate(val_loader, model, loss_function):
    """
    Defines the behaviour of validation - 1 epoch.
        val_loader     - DataLoader of the Class TrainDataset
        model          - SSD network
        loss_function: - MultiBox loss
    """

    model.eval()

    losses = AverageMeter()

    with torch.no_grad():
        for i, (images, boxes, labels) in enumerate(val_loader):

            #To CUDA - GPU training
            images = images.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            #Get Predictions out of the model - Forward Propagation
            predicted_locs, predicted_scores = model(images)

            #Compute loss using MultiBoxLoss function
            loss = loss_function(predicted_locs, predicted_scores, boxes, labels)

            losses.update(loss.item(), images.size(0))

    print('\n * Validation AVG LOSS - {loss.avg:.3f}\n'.format(loss=losses))

    return losses

def save_best_trained(epoch, epochs_since_improvement, model, optimizer, best_loss, training_losses, validation_losses):
    """
    Save best model.
    :epoch: epoch number
    :epochs_since_improvement: number of epochs since last improvement
    :model: model
    :optimizer: optimizer
    :loss: validation loss in this epoch
    """
    t_loss_normal, t_loss_avg = training_losses[0], training_losses[1]
    v_loss_normal, v_loss_avg = validation_losses[0], validation_losses[1]
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'loss': best_loss,
             'training_losses_batch_values' : t_loss_normal,
             'training_losses_batch_avgs' : t_loss_avg,
             'validation_losses_batch_values' : v_loss_normal,
             'validation_losses_batch_avgs' : v_loss_avg,
             'model_state_dict': model.state_dict(),
             'optimizer': optimizer}
    filename = 'model_ssd300.pth.tar'
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    torch.save(state, './saved_models/BEST_' + filename)
