import sim
import pybullet as p
import random
import numpy as np
import math
import argparse
from dataset import PointCloudDataset

#Continuous Play: 

#model
    #input: voxel-representation of object
    #output: voxels where object can be grasped. 
    #loss: 

# for each object, randomly place it in bin
# sample pickup 
# label pickup success or failure 
# update model 

#Questions: 
    #best practices for SSL in robotics?
    #when/how to end simulation?
    #what criteria to end on?
    
from math import atan2
from turtle import right
from typing import Dict, List, Tuple

import os
import argparse
from functools import lru_cache
from random import seed
import json
from matplotlib.pyplot import yscale

import numpy as np
from skimage.io import imsave
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from image import read_rgb
import affordance_model
from common import save_chkpt, load_chkpt

@lru_cache(maxsize=128)
def read_rgb_cached(file_path):
    return read_rgb(file_path)

def train(model, train_loader, criterion, optimizer, epoch, device):
    """
        Loop over each sample in the dataloader. Do forward + backward + optimize procedure and print mean IoU on train set.
        :param model (torch.nn.module object): miniUNet model object
        :param train_loader (torch.utils.data.DataLoader object): train dataloader 
        :param criterion (torch.nn.module object): Pytorch criterion object
        :param optimizer (torch.optim.Optimizer object): Pytorch optimizer object
        :param epoch (int): current epoch number
        :return mean_epoch_loss (float): mean loss across this epoch
    """
    
    #input voxel representation to model 
    #match model to 
    
    return 


def test(model, test_loader, criterion, device, save_dir=None):
    """
        Similar to train(), but no need to backward and optimize.
        :param model (torch.nn.module object): miniUNet model object
        :param test_loader (torch.utils.data.DataLoader object): test dataloader 
        :param criterion (torch.nn.module object): Pytorch criterion object
        :return mean_epoch_loss (float): mean loss across this epoch
        :return mean_iou (float): mean iou across this epoch
    """
    

    return 


def save_prediction(
        model: torch.nn.Module, 
        dataloader: DataLoader, 
        dump_dir: str, 
        BATCH_SIZE:int
    ) -> None:
    print(f"Saving predictions in directory {dump_dir}")
    


def main():
    parser = argparse.ArgumentParser(description='Model training script')
    parser.add_argument('-m', '--model', default='affordance',
        help='which model to train: "affordance" or "action_regression"')
    parser.add_argument('-a', '--augmentation', action='store_true',
        help='flag to enable data augmentation')
    args = parser.parse_args()


    # Check if GPU is being detected
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

if __name__ == "__main__":
    main()
