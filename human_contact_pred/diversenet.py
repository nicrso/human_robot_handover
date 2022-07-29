from operator import pos
from matplotlib import image, transforms 
import torch 
import numpy as np
from torch import nn
import torch.nn.functional as F
from typing import Tuple, List

import wandb

from losses import DiverseLoss
from visualize import process_voxels, save_preds

import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import ModelSummary

import open3d as o3d
from open3d.visualization.tensorboard_plugin import summary
from open3d.visualization.tensorboard_plugin.util import to_dict_batch
from torch.utils.tensorboard import SummaryWriter
import torchvision
import sys 
from utils import reshape_data_for_logging

np.set_printoptions(threshold=sys.maxsize)

class VoxNet(pl.LightningModule):
    """
    3D CNN as described in "VoxNet: A 3D Convolutional Neural Network for Real-Time Object
    Recognition" -- Daniel Maturana and Sebastian Scherer
    """

    def __init__(
        self, 
        n_ensemble: int = 10, 
        in_channels: int = 5,
        out_channels: int = 2,
        droprate=0
    ) -> None:
        """
        3D CNN PyTorch Lightning module. 

        :param n_ensemble: Number of models to train 
        :param inplanes: Number of input features
        :param outplanes: Number of output features (classes)
        :param droprate: Probability of dropping a prediction
        """

        super(VoxNet, self).__init__()

        self.droprate = droprate
        self.drop = nn.Dropout(p=droprate)

        nc = in_channels

        self.down1 = nn.Sequential(
            self.conv_block(in_channels, nc*4), 
            nn.MaxPool3d(2),
        )
        nc *= 4
    	
        self.down2 = nn.Sequential(
            self.conv_block(nc+n_ensemble, nc*4),
            nn.MaxPool3d(2),
        )
        nc *= 4

        self.down3 = nn.Sequential(
            self.conv_block(nc, nc*4),
            nn.MaxPool3d(4),
        )
        nc *= 4

        self.up1 = nn.Sequential(
            self.conv_block(nc, nc//4),
        )
        nc = nc // 4

        self.up2 = nn.Sequential(
            self.conv_block(nc, nc//4),
        )
        nc = nc // 4

        self.up3 = nn.Sequential(
            self.conv_block(nc+n_ensemble, nc//4),
        )
        nc = nc // 4

        self.up4 = nn.Conv3d(nc, out_channels, kernel_size=3, padding=1)

        #Look into why this is done in Voxnet paper 
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    
    @staticmethod
    def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        """
        Repeatable convolutional block. 

        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :return: The sequential convolutional block
        """

        return nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm3d(out_channels),
        )

    def forward(self, x: torch.Tensor, c) -> torch.Tensor:
        """
        Forward defines prediction/inference actions.

        :param x: Input tensor 
        :param c:
        :return: Model output tensor
        """
        #lightning: forward defines prediction/inference actions
        if self.droprate > 0:
            x = self.drop(x)

        #Encoding pass
        x = self.down1(x)
        x = torch.cat((x, c), dim=1)
        x = self.down2(x)
        x = self.down3(x)
        
        #Decoding pass
        x = x.view(x.shape[0], -1, 4, 4, 4)
        x = F.interpolate(self.up1(x), scale_factor=4)
        x = F.interpolate(self.up2(x), scale_factor=2)
        x = torch.cat((x, c), dim=1)
        x = F.interpolate(self.up3(x), scale_factor=2)
        x = self.up4(x)

        return x
    
class DiverseVoxNet(pl.LightningModule):
    """
    Generalization of Voxnet to DiverseNet as specified 
    by "DiverseNet: When One Right Answer is not Enough"
    -- Firman et. al. 
    """

    def __init__(
        self, 
        n_ensemble: int = 10,
        in_channels: int = 5,
        out_channels: int = 2, 
        droprate: int = 0,
        diverse_beta: int = 1,
        learning_rate: float = 1e-3,
        weight_decay: float = 5e-4, 
        momentum: float = 0.9,
        lr_gamma: float = 0.1, 
        lr_step_size: int = 1000, 
    ) -> None: 
        """
        3D CNN DiverseNet PyTorch Lightning module. 

        :param n_ensemble: Number of models to train 
        :param inplanes: Number of input features
        :param outplanes: Number of output features (classes)
        :param droprate: Probability of dropping a prediction
        :param diverse_beta: beta for loss
        """
        super(DiverseVoxNet, self).__init__()

        self.n_ensemble = n_ensemble
        self.voxnet = VoxNet(
            n_ensemble=n_ensemble,
            in_channels=in_channels,
            droprate=droprate,
        )

        self.train_loss_fn = DiverseLoss(beta=diverse_beta)
        self.val_loss_fn = DiverseLoss(beta=diverse_beta, is_train=False)
        self.accuracy = DiverseLoss(beta=diverse_beta, eval_mode=True)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay	
        self.momentum = momentum
        self.lr_gamma = lr_gamma
        self.lr_step_size = lr_step_size

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """
        :param xs: N x 5 x DHW
        :return: N x Ensemble X DHW
        """

        N, _, D, H, W = xs.shape
        preds = []

        for x in xs:
            x = x.view(1, *x.shape).expand(self.n_ensemble, -1, -1, -1, -1)
            c = torch.eye(self.n_ensemble, dtype=x.dtype, device=x.device)
            c = c.view(*c.shape, 1, 1, 1).expand(-1, -1, D//2, H//2, W//2)
            pred = self.voxnet(x, c)
            preds.append(pred)
        preds = torch.stack(preds)
        return preds

    def training_step(
        self, 
        batch: Tuple[torch.Tensor, torch.Tensor], 
        batch_idx: int,
    ) -> float:
        """
        Returns the training loss and logs loss.

        :param batch: The input and target training data
        :param batch_idx: The index of the given batch
        :return: The training loss
        """

        geom, tex_targs = batch 
        tex_preds = self(geom)

        loss, _ = self.train_loss_fn(tex_preds, tex_targs)
        acc, _ = self.accuracy(tex_preds, tex_targs)
        acc = 1 - acc

        plot_type = "pointcloud"

        # print("Training")
        # print(geom.shape, tex_preds.shape, tex_targs.shape)
        #torch.Size([6, 5, 64, 64, 64]) torch.Size([6, 6, 2, 64, 64, 64]) torch.Size([6, 6, 64, 64, 64])

        if (self.current_epoch + 1) % 100 == 0 or self.current_epoch == 0:
            im_name = "Epoch_" + str(self.current_epoch) + "_Step_" + str(self.global_step) + '_Train_' + str(plot_type) + ".png"
            
            geom, tex_preds, tex_targs = reshape_data_for_logging(geom, tex_preds, tex_targs)

            #save images
            # trans = torchvision.transforms.ToTensor()
            # img = trans(save_preds(geom[0], tex_preds, im_name, plot_type, True, tex_targs, "diversenet"))

            #save predictions 
            xyz_arr = []

            xyz_targ, c_targ = process_voxels(geom[0], tex_targs[0], is_pred=False)
            xyz_targ = np.concatenate((xyz_targ, 255*c_targ), axis=1)
            xyz_arr.append(wandb.Object3D(xyz_targ))

            for tex_pred in tex_preds:
                xyz_pred, c_pred = process_voxels(geom[0], tex_pred, is_pred=True)
                xyz_pred = np.concatenate((xyz_pred, 255*c_pred), axis=1)
                xyz_arr.append(wandb.Object3D(xyz_pred))
            
            wandb_logger = self.logger.experiment

            wandb_logger.log({"train_point_clouds": xyz_arr})          
            
        #plot accuracy too 

        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True)

        return loss
    
    def validation_step(
        self, 
        batch: Tuple[torch.Tensor, torch.Tensor], 
        idx: int,
    ) -> float:
        """
        Returns the validation loss. 

        :param batch: The input and target validation data
        :param idx: Current batch index
        :return: The validation loss
        """

        geom, tex_targs = batch 
        tex_preds = self(geom)

        loss, _ = self.val_loss_fn(tex_preds, tex_targs)
        acc, _ = self.accuracy(tex_preds, tex_targs)
        acc = 1 - acc

        # print("Validation")
        # print(geom.shape, tex_preds.shape, tex_targs.shape)
        # torch.Size([3, 5, 64, 64, 64]) torch.Size([3, 6, 2, 64, 64, 64]) torch.Size([3, 6, 64, 64, 64])

        plot_type = "pointcloud"

        if (self.current_epoch + 1) % 100 == 0 or self.current_epoch == 0:
            im_name = "Epoch_" + str(self.current_epoch) + "_Step_" + str(self.global_step) + '_Val_' + str(plot_type) + ".png"
            
            geom, tex_preds, tex_targs = reshape_data_for_logging(geom, tex_preds, tex_targs, train=False)

            wandb_logger = self.logger.experiment

            for i in range(geom.shape[0]):

                #actual: (5, 64, 64, 64) (6, 2, 64, 64, 64) (6, 64, 64, 64)
                #expected (5, 64, 64, 64) (6, 2, 64, 64, 64) (3, 6, 64, 64, 64)

                #save images
                # trans = torchvision.transforms.ToTensor()
                # img = trans(save_preds(geom[0], tex_preds, im_name, plot_type, True, tex_targs, "diversenet"))

                #save target & prediction meshes
                xyz_arr = []

                xyz_targ, c_targ = process_voxels(geom[i][0], tex_targs[i], is_pred=False)
                xyz_targ = np.concatenate((xyz_targ, 255*c_targ), axis=1)
                xyz_arr.append(wandb.Object3D(xyz_targ))

                for tex_pred in tex_preds[i]:
                    xyz_pred, c_pred = process_voxels(geom[i][0], tex_pred, is_pred=True)
                    xyz_pred = np.concatenate((xyz_pred, 255*c_pred), axis=1)
                    xyz_arr.append(wandb.Object3D(xyz_pred))
                
                
                wandb_logger.log({"val_point_clouds": xyz_arr})    

        self.log("val_acc", acc, on_step=False, on_epoch=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configures optimizers for the model.
        :return: Configured optimizers 
        """

        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        #Note: might also try SGD, but research shows fine-tuned adam outperforms 

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma)

        return [optimizer], [lr_scheduler]

if __name__ == "__main__":
    model = DiverseVoxNet()
    print(model)