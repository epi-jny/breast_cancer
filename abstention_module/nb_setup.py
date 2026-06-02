# nb_setup.py

# --- Standard library ---
import os, sys, math, pickle, warnings, random as rd, ast
from typing import Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path

# --- Scientific stack ---
import numpy as np
import pandas as pd
import scipy.special
from scipy.special import gammaln
from sklearn.model_selection import train_test_split


# --- Plotting ---
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.ticker import AutoMinorLocator
from matplotlib import MatplotlibDeprecationWarning
from matplotlib import lines

# --- PyTorch / TorchVision ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

from torch.utils.data import (
    random_split,
    DataLoader,
    ConcatDataset,
    WeightedRandomSampler,
)

from torchvision import datasets, transforms, models
from torchvision.models import VGG16_Weights, ResNet18_Weights, resnet18

# --- Utilities ---
from tqdm import tqdm
from IPython.display import clear_output

# --- Project modules (support .. and ../..) ---
current_dir = os.getcwd()

root_path = os.path.abspath(os.path.join(current_dir, '..', '..'))
if root_path not in sys.path:
    sys.path.append(root_path)

module_path = os.path.abspath(os.path.join(current_dir, '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from abstention_module.sgp_utils import *
from abstention_module.preprocessing import *
from abstention_module.mcdropout import *     # MC Dropout utilities
from abstention_module.plotting import *
from abstention_module.math_utils import *
from abstention_module import plotting

# --- Config ---
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
mpl.rcParams['figure.dpi'] = 150

# --- GPU info ---
print("GPU Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
