
import os 
import configparser
import argparse

import torch 
from torch import embedding, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from voxel_dataset import VoxelDataset
from Robotics.human_contact_pred.diversenet import DiverseVoxNet, VoxNet

osp = os.path

def eval(
    data_dir,
    instruction,
    checkpoint_path,
    config_filename,
    test_only=False,
    show_object=None,
    save_preds=False,
):
    #config
    config = configparser.ConfigParser()
    config.read(config_filename)
    droprate = config['hyperparams'].getfloat('droprate')

    #load model from checkpoint
    model = DiverseVoxNet.load_from_checkpoint(checkpoint_path)

    #figure out how to use this 
    #kwargs = dict(data_dir=data_dir, instruction=instruction, train=False, random_rotation=0, n_ensemble=-1, test_only=test_only)

    #print model hyperparams
    checkpoint = torch.load(checkpoint_path)
    print(checkpoint["hyper_parameters"])

    #disable randomness, dropout, etc. 
    model.eval()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=osp.join('data', 'voxelized_meshes'))
    parser.add_argument('--instruction', required=True)
    parser.add_argument('--checkpoint_filename', required=True)
    parser.add_argument('--config_filename', required=True)
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--show_object', default=None)
    parser.add_argument('--save_preds', default=False)
    args = parser.parse_args()

    eval(
        osp.expanduser(args.data_dir), 
        args.instruction,
        osp.expanduser(args.checkpoint_filename),
        osp.expanduser(args.config_filename),
        test_only=args.test_only, 
        show_object=args.show_object, 
        save_preds=args.save_preds
        )
