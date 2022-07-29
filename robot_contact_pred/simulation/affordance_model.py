from typing import Tuple, Optional, Dict

import numpy as np
from matplotlib import cm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

from common import draw_grasp


def get_gaussian_scoremap(
        shape: Tuple[int, int], 
        keypoint: np.ndarray, 
        sigma: float=1, dtype=np.float32) -> np.ndarray:
    """
    Generate a image of shape=:shape:, generate a Gaussian distribtuion
    centered at :keypont: with standard deviation :sigma: pixels.
    keypoint: shape=(2,)
    """
    coord_img = np.moveaxis(np.indices(shape),0,-1).astype(dtype)
    sqrt_dist_img = np.square(np.linalg.norm(
        coord_img - keypoint[::-1].astype(dtype), axis=-1))
    scoremap = np.exp(-0.5/np.square(sigma)*sqrt_dist_img)
    return scoremap

class AffordanceDataset(Dataset):
    """
    Transformational dataset.
    raw_dataset is of type train.RGBDataset
    """
    def __init__(self, raw_dataset: Dataset):
        super().__init__()
        self.raw_dataset = raw_dataset
    
    def __len__(self) -> int:
        return len(self.raw_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Transform the raw RGB dataset element into
        training targets for AffordanceModel.
        return: 
        {
            'input': torch.Tensor (3,H,W), torch.float32, range [0,1]
            'target': torch.Tensor (1,H,W), torch.float32, range [0,1]
        }
        Note: self.raw_dataset[idx]['rgb'] is torch.Tensor (H,W,3) torch.uint8
        """
        # checkout train.RGBDataset for the content of data
        data = self.raw_dataset[idx]
        # TODO: complete this method
        # Hint: Use get_gaussian_scoremap
        # Hint: https://imgaug.readthedocs.io/en/latest/source/examples_keypoints.html
        # ===============================================================================
        
        img = data['rgb'].numpy()
        cntr = data['center_point']
        ang = data['angle'].item()
                
        img_shape=np.shape(img)[:2]
        
        kps = KeypointsOnImage([
            Keypoint(x=cntr[0], y=cntr[1])
        ], shape=img_shape)
        
        rot = iaa.Rotate(-float(ang))
        img_aug, kps_aug = rot(image=img, keypoints=kps)
        
        img_aug = np.moveaxis(img_aug, -1, 0)
        img_aug = torch.squeeze(torch.from_numpy(img_aug/255).to(torch.float32))
        # print('img', img_aug.size())
        kps_aug = np.array([kps_aug.keypoints[0].x, kps_aug.keypoints[0].y])
        guass = torch.from_numpy(get_gaussian_scoremap(img_shape, kps_aug)).to(torch.float32).unsqueeze(0)
        # print('guass', guass.size())
        
        return dict(input=img_aug, target=guass)
        # ===============================================================================


class AffordanceModel(nn.Module):
    def __init__(self, n_channels: int=3, n_classes: int=1, **kwargs):
        """
        A simplified U-Net with twice of down/up sampling and single convolution.
        ref: https://arxiv.org/abs/1505.04597, https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
        :param n_channels (int): number of channels (for grayscale 1, for rgb 3)
        :param n_classes (int): number of segmentation classes (num objects + 1 for background)
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = nn.Sequential(
            # For simplicity:
            #     use padding 1 for 3*3 conv to keep the same Width and Height and ease concatenation
            nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential( 
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.outc = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1)
        # hack to get model device
        self.dummy_param = nn.Parameter(torch.empty(0))

    @property
    def device(self) -> torch.device:
        return self.dummy_param.device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_inc = self.inc(x)  # N * 64 * H * W
        x_down1 = self.down1(x_inc)  # N * 128 * H/2 * W/2
        x_down2 = self.down2(x_down1)  # N * 256 * H/4 * W/4
        x_up1 = self.upconv1(x_down2)  # N * 128 * H/2 * W/2
        x_up1 = torch.cat([x_up1, x_down1], dim=1)  # N * 256 * H/2 * W/2
        x_up1 = self.conv1(x_up1)  # N * 128 * H/2 * W/2
        x_up2 = self.upconv2(x_up1)  # N * 64 * H * W
        x_up2 = torch.cat([x_up2, x_inc], dim=1)  # N * 128 * H * W
        x_up2 = self.conv2(x_up2)  # N * 64 * H * W
        x_outc = self.outc(x_up2)  # N * n_classes * H * W
        return x_outc

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict affordance using this model.
        This is required due to BCEWithLogitsLoss
        """
        return torch.sigmoid(self.forward(x))

    @staticmethod
    def get_criterion() -> torch.nn.Module:
        """
        Return the Loss object needed for training.
        Hint: why does nn.BCELoss does not work well?
        """
        return nn.BCEWithLogitsLoss()

    @staticmethod
    def visualize(input: np.ndarray, output: np.ndarray, 
            target: Optional[np.ndarray]=None) -> np.ndarray:
        """
        Visualize rgb input and affordance as a single rgb image.
        """
        cmap = cm.get_cmap('viridis')
        in_img = np.moveaxis(input, 0, -1)
        pred_img = cmap(output[0])[...,:3]
        row = [in_img, pred_img]
        if target is not None:
            gt_img = cmap(target[0])[...,:3]
            row.append(gt_img)
        img = (np.concatenate(row, axis=1)*255).astype(np.uint8)
        return img


    def predict_grasp(self, rgb_obs: np.ndarray
            ) -> Tuple[Tuple[int, int], float, np.ndarray]:
        """
        Given a RGB image observation, predict the grasping location and angle in image space.
        return coord, angle, vis_img
        :coord: tuple(int x, int y). By OpenCV convension, x is left-to-right and y is top-to-bottom.
        :angle: float. By OpenCV convension, angle is clockwise rotation.
        :vis_img: np.ndarray(shape=(H,W,3), dtype=np.uint8). Visualize prediction as a RGB image.

        Note: torchvision's rotation is counter clockwise, while imgaug,OpenCV's rotation are clockwise.
        """
        device = self.device
        # TODO: complete this method (prediction)
        
        #take rgb input
        #rotate to [0, 7]*22.5
        #stack batchwise
        rotations = [i*22.5 for i in range(8)]
        img_shape = np.array(rgb_obs).shape
        input = []
        
        for i in rotations:
            aug = iaa.Rotate(-1*i)
            img_aug = aug(image=rgb_obs) 
            img_aug = torch.unsqueeze(torch.from_numpy(np.moveaxis(img_aug/255, -1, 0)).to(torch.float32), 0)
            input.append(img_aug)
            
            
        #feed into network
        data = torch.cat(input, dim=0)
        
        # print("shape", input.shape)
        data = data.to(device)
        pred = self.predict(data).to(device)
        
        # print("output shape", output.size())
        
        #find the max affordance pixel accross all 8 images
        amax = torch.argmax(pred).cpu()
        # print("AMAX", amax)
        amax = np.unravel_index(amax, pred.shape)
        # print("AMAX COORD", amax)
        angle_index = amax[0].item()
        
        # Hint: why do we provide the model's device here?
        # ===============================================================================
        coord, angle = tuple([int(amax[3].item()), int(amax[2].item())]), rotations[angle_index]
        
        # TODO: complete this method (visualization)
        # :vis_img: np.ndarray(shape=(H,W,3), dtype=np.uint8). Visualize prediction as a RGB image.
        
        vis_lst = []
        
        for i in range(0, 8):
            #pair image with its predicted target
            img = data[i].detach().cpu().numpy()
            predict = pred[i].detach().cpu().numpy()
            img_couple = self.visualize(img, predict)
            if i == angle_index:
                draw_grasp(img_couple, coord, 0)
            img_couple[127, :, :] = 127
            vis_lst.append(img_couple)
        
        row_list = []
        for i in range(0, 4):
            img1 = vis_lst[2*i]
            img2 = vis_lst[(2*i)+1]     
            row_list.append(np.concatenate((img1, img2), axis=1))
        
        vis_img = np.concatenate(row_list, axis=0)
        
        # Hint: use common.draw_grasp
        # Hint: see self.visualize
        # Hint: draw a grey (127,127,127) line on the bottom row of each image.
        # ===============================================================================
        rot = iaa.Rotate(rotations[angle_index])
        kps = KeypointsOnImage([
            Keypoint(
                    x=coord[0], 
                    y=coord[1]
                )
        ], shape=rgb_obs.shape)
        kps_aug = rot(keypoints=kps)
        coord = (int(kps_aug[0].x), int(kps_aug[0].y))
        # ===============================================================================
        return coord, angle, vis_img

