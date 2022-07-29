import os 
import configparser
import argparse
from sklearn.model_selection import learning_curve

import torch 
import wandb
from torch import embedding, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from zmq import device

from voxel_dataset import VoxelDataset
from diversenet import DiverseVoxNet, VoxNet
from cnn3d import cnn3d
from pytorch_lightning.loggers import WandbLogger

osp = os.path

 
def train(
    data_dir, 
    instruction, 
    config_file, 
    model_type,
    checkpoint_file=None,
    experiment_suffix=None,
    include_sessions=None,
    exclude_sessions=None
):

    #Config
    config = configparser.ConfigParser()
    config.read(config_file)

    section = config['optim']
    batch_size = section.getint('batch_size')
    max_epochs = section.getint('max_epochs')
    val_interval = section.getint('val_interval')
    do_val = val_interval > 0
    base_lr = section.getfloat('base_lr')
    momentum = section.getfloat('momentum')
    weight_decay = section.getfloat('weight_decay')

    section = config['misc']
    log_interval = section.getint('log_interval')
    shuffle = section.getboolean('shuffle')
    num_workers = section.getint('num_workers')
    logger_type = section['logger']

    section = config['hyperparams']
    n_ensemble = section.getint('n_ensemble')
    diverse_beta = section.getfloat('diverse_beta')
    pos_weight = section.getfloat('pos_weight')
    droprate = section.getfloat('droprate')
    lr_step_size = section.getint('lr_step_size', 1000)
    lr_gamma = section.getfloat('lr_gamma', 0.1)
    grid_size = section.getint('grid_size')
    random_rotation = section.getfloat('random_rotation')

    resume = False if checkpoint_file is None else True

    kwargs = dict(
        data_dir=data_dir, 
        instruction=instruction,
        include_sessions=include_sessions, 
        exclude_sessions=exclude_sessions,
        n_ensemble=n_ensemble
        )

    #Training dataset
    train_dset = VoxelDataset(
        grid_size=grid_size,
        random_rotation=random_rotation,
        is_train=True,
        **kwargs
    )
    train_loader = DataLoader(
        train_dset, 
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True, 
        num_workers=num_workers
    )

    #Val dataset
    val_dset = VoxelDataset(
        grid_size=grid_size,
        random_rotation=0,
        is_train=False,
        **kwargs
    )
    val_loader = DataLoader(
        val_dset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers
    )

    #Create model
    diverse_voxnet = DiverseVoxNet(
        n_ensemble=n_ensemble,
        droprate=droprate,
        diverse_beta=diverse_beta,
        learning_rate=base_lr,
        weight_decay=weight_decay, 
        momentum=momentum,
        lr_gamma=lr_gamma,
        lr_step_size=lr_step_size,
    )
    
    single_voxnet = cnn3d()

    if model_type == 'single':
        voxnet = single_voxnet
    elif model_type == 'diverse':
        voxnet = diverse_voxnet

    #checkpointing
    checkpoint_callback = ModelCheckpoint(save_top_k=2, monitor="val_loss", mode="min", dirpath="data/checkpoints", filename="{model_type}-{epoch}-{val_loss:.5f}", every_n_epochs=1)
    #trainer = pl.Trainer(devices=1, accelerator="gpu", callbacks=[checkpoint_callback])

    trainer = pl.Trainer(accelerator="gpu", devices=1)

    print("Logger: ", logger_type)

    #init tensorboard logger
    if logger_type == "tb":
        logger = TensorBoardLogger("tb_logs", name="voxnet")
        trainer = pl.Trainer(accelerator='gpu', devices=1, logger=logger, callbacks=[checkpoint_callback])

    #init wandb logger 
    elif logger_type == "wandb":
        logger = WandbLogger(project="human-contact-prediction")
        trainer = pl.Trainer(accelerator='gpu', devices=1, logger=logger, callbacks=[checkpoint_callback])

    if resume:
        trainer.fit(
            model=voxnet, 
            train_dataloaders=train_loader, 
            val_dataloaders=val_loader, 
            ckpt_path=checkpoint_file
        )
    else: 
        trainer.fit(
            model=voxnet, 
            train_dataloaders=train_loader, 
            val_dataloaders=val_loader
        )

    #test using best checkpoint 
    trainer.test(ckpt_path="best")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
        default=osp.join('data', 'voxelized_meshes'))
    parser.add_argument('--instruction', required=True)
    parser.add_argument('--config_file', required=True)
    parser.add_argument('--model_type', required=True) #single or diverse
    parser.add_argument('--suffix', default=None)
    parser.add_argument("--checkpoint_file", default=None)
    parser.add_argument('--include_sessions', default=None)
    parser.add_argument('--exclude_sessions', default=None)
    args = parser.parse_args()

    include_sessions = None
    if args.include_sessions is not None:
        include_sessions = args.include_sessions.split(',')
    exclude_sessions = None
    if args.exclude_sessions is not None:
        exclude_sessions = args.exclude_sessions.split(',')
    train(osp.expanduser(args.data_dir), args.instruction, args.config_file, args.model_type,
        experiment_suffix=args.suffix,checkpoint_file=args.checkpoint_file, include_sessions=include_sessions,
        exclude_sessions=exclude_sessions)