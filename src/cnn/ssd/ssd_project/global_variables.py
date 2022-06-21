import torch
import torch.backends.cudnn as cudnn
import os
# Labels map
LABEL_MAP = {'background'  : 0,
              'building'   : 1,
              'window'     : 2,
              'door'       : 3}

REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}  # Inverse mapping
COLORS_CLASSES = [(255, 255, 0),(60, 180, 75), (230, 25, 75), (0,0,0)]
LABEL_COLOR_MAP = {k: COLORS_CLASSES [i] for i, k in enumerate(LABEL_MAP.keys())}
NUM_CLASSES = len(LABEL_MAP)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRE_TRAINED_MODEL = None
BATCH_SIZE = 8
START_EPOCH = 0
EPOCHS = 400
EPOCHS_SINCE_IMPROVEMENT = 0
BEST_LOSS = 100. #Start with big loss
WORKERS = 4 # number of workers in DataLoader
LEARNING_RATE = 1e-3  # learning rate
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
cudnn.benchmark = True
SPLIT_RATIO = 0.7

#VGG16_WEIGHTS_PRETRAINED = torch.load("weights/" + "vgg16_reducedfc.pth")
print(os.getcwd())
VGG16_WEIGHTS_PRETRAINED = torch.load("cnn/ssd/weights/" + "vgg16_reducedfc.pth")
#VGG16_WEIGHTS_PRETRAINED = torch.load("../weights/" + "vgg16_reducedfc.pth")
