from ast import Div
import os
import numpy 
import torch 
from torch import nn
import torch.nn.functional as F
from typing import Tuple, List

from losses import DiverseLoss, TextureLoss
from visualize import save_preds

import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import ModelSummary
import torchvision

#alter model to handle the format of the dataloader 
#dataloader items will be np arrays with one item 

class cnn3d(pl.LightningModule):
    """
    3D CNN as described in "VoxNet: A 3D Convolutional Neural Network for Real-Time Object
    Recognition" -- Daniel Maturana and Sebastian Scherer
    """

    def __init__(
        self, 
        in_channels: int = 5,
        out_channels: int = 2,
        diverse_beta: int = 1,
        learning_rate: float = 1e-3,
        weight_decay: float = 5e-4, 
        momentum: float = 0.9,
        lr_gamma: float = 0.1, 
        lr_step_size: int = 1000, 
    ) -> None:
        """
        3D CNN PyTorch Lightning module. 
        :param in channels: Number of input features
        :param out channels: Number of output features (classes)
        """

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay	
        self.momentum = momentum
        self.lr_gamma = lr_gamma
        self.lr_step_size = lr_step_size

        super(cnn3d, self).__init__()

        #initialize losses 
        self.train_loss_fn = DiverseLoss()
        self.val_loss_fn = DiverseLoss(is_train=False)

        nc = in_channels

        self.inc = nn.Sequential(
            self.conv_block(in_channels, nc*2) 
        )
        nc *= 2

        self.inc2 = nn.Sequential(
            self.conv_block(nc, nc*2) 
        )
        nc *= 2

        self.down1 = nn.Sequential(
            nn.MaxPool3d(2),
            self.conv_block(nc, nc*2), 
        )
        nc *= 2

        self.down2 = nn.Sequential(
            nn.MaxPool3d(2),
            self.conv_block(nc, nc*2), 
        )
        nc *= 2

        self.down3 = nn.Sequential(
            nn.MaxPool3d(2),
            self.conv_block(nc, nc*2), 
        )
        nc *= 2

        self.down4 = nn.Sequential(
            nn.MaxPool3d(2),
            self.conv_block(nc, nc*2)
        )
        nc *= 2

        self.up1 = nn.ConvTranspose3d(nc, nc//2, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            self.conv_block(nc, nc//2), 
        )
    
        nc = nc // 2

        self.up2 = nn.ConvTranspose3d(nc, nc//2, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            self.conv_block(nc, nc//2), 
        )
        
        nc = nc // 2

        self.up3 = nn.ConvTranspose3d(nc, nc//2, kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            self.conv_block(nc, nc//2), 
        )
        
        nc = nc // 2

        self.up4 = nn.ConvTranspose3d(nc, nc//2, kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(
            self.conv_block(nc, nc//2), 
        )

        nc = nc // 2

        self.conv5 = nn.Conv3d(nc, out_channels, kernel_size=3, padding=1) 


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward defines prediction/inference actions.
        :param x: Input tensor - 5 * 64 * 64 * 64
        :return: Model output tensor
        """

        #Batch_Size = B

        #Encoding pass

        x0 = self.inc(x)
        x1 = self.inc2(x0) # B * 10 * 64 * 64 * 64
        x2 = self.down1(x1) # B * 20 * 32 * 32 * 32
        x3 = self.down2(x2) # B * 40 * 16 * 16 * 16
        x4 = self.down3(x3) # B * 80 * 8 * 8 * 8
        x5 = self.down4(x4) #B * 160 * 4 * 4 * 4

        # #Decoding pass

        x = self.up1(x5) 
        x = torch.cat([x, x4], dim=1)
        x = self.conv1(x) #B * 80 * 8 * 8 * 8

        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv2(x) #B * 40 * 16 * 16 * 16

        x = self.up3(x) 
        x = torch.cat([x, x2], dim=1) 
        x = self.conv3(x) #B * 20 * 32 * 32 * 32

        x = self.up4(x) 
        x = torch.cat([x, x1], dim=1) 
        x = self.conv4(x) 

        x = self.conv5(x) 

        return x

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

        tex_preds = torch.unsqueeze(tex_preds, dim=0)

        loss, _ = self.train_loss_fn(tex_preds, tex_targs)
        self.log("train_loss", loss, on_step=False, on_epoch=True)

        plot_type = "pointcloud"

        if (self.current_epoch+1) % 50 == 0 or self.current_epoch == 0:
            im_name = "Epoch_" + str(self.current_epoch) + "_Step_" + str(self.global_step) + '_' + str(plot_type) + ".png"
            
            if geom.shape[0] > 1:
                geom = geom[0]
            geom = geom.detach().cpu().numpy().squeeze()
 
            if tex_preds.shape[0] > 1:
                tex_preds = tex_preds[0]
            tex_preds = tex_preds.detach().cpu().numpy()

            tex_targs = tex_targs.detach().cpu().numpy()
            
            trans = torchvision.transforms.ToTensor()

            tex_preds = tex_preds[0]

            if tex_preds.shape[0] != 1:
                tex_preds = tex_preds[0]
                tex_preds = numpy.expand_dims(tex_preds, 0)

            img = trans(save_preds(geom[0], tex_preds, im_name, plot_type, True, tex_targs, "cnn3d"))

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

        tex_preds = torch.unsqueeze(tex_preds, dim=0)

        loss, _ = self.val_loss_fn(tex_preds, tex_targs)

        self.log("val_loss", loss, on_step=False, on_epoch=True)

        # plot_type = "pointcloud"

        # if self.current_epoch % 100 == 0:
        #     im_name = "Epoch_" + str(self.current_epoch) + "_Step_" + str(self.global_step) + '_' + str(plot_type) + ".png"
            
        #     if geom.shape[0] > 1:
        #         geom = geom[0]
        #     geom = geom.cpu().numpy().squeeze()
 
        #     if tex_preds.shape[0] > 1:
        #         tex_preds = tex_preds[0]

        #     tex_preds = tex_preds.cpu().numpy().squeeze()
        #     tex_targs = tex_targs.cpu().numpy()
            
        #     trans = torchvision.transforms.ToTensor()

        #     if geom[0].shape[0] == 5:
        #         geom = geom[0]

        #     if tex_preds.shape[0] == 2:
        #         tex_preds = numpy.expand_dims(tex_preds, 0)

        #     #save_preds expects: geom.shape(5, 64, 64, 64), tex_preds.shape(X, 2, 64, 64, 64)

        #     img = trans(save_preds(geom[0], tex_preds, im_name, plot_type, True, tex_targs, "cnn3d"))

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
    model = cnn3d()
    print(model) 