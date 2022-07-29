import visualize
from torch import float64, threshold
import torch
import utils
from voxel_dataset import VoxelDataset
from losses import DiverseLoss, TextureLoss
import os
import os.path as osp
import sys 
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

def create_test_meshes(dset, idx=14):
    
    geom, tex_targ = dset[idx]

    z, y, x = np.nonzero(geom[0])  # see which voxels are occupied
    c = tex_targ[0, z, y, x]

    xyz = np.vstack((x, y, z)).T

    utils.show_pointcloud(xyz, c)

    #show all the interior points 
    cmap = np.asarray([0, 0, 1])
    c_int = cmap[c]

    utils.show_pointcloud(xyz, c_int)

    # make all pred test target array 
    cmap = np.asarray([1, 1, 1])
    c_full = cmap[c]

    utils.show_pointcloud(xyz, c_full)

    tex_targ[0, z, y, x] = c_full
    
    tex_targs = np.expand_dims(tex_targ, axis=0)

    #make test pred array 

    tex_preds = np.full((2, 64, 64, 64), 1, dtype=np.float32)

    #randomly assigns colors 
    mask = np.random.randint(0,2,size=tex_preds.shape).astype(np.bool)
    r = np.full(shape=tex_preds.shape, fill_value=10, dtype=np.float32)
    tex_preds[mask] = r[mask]

    # tex_preds[0, :, :, :] = 10 #makes everything blue
    # tex_preds[1, :, :, :] = 10 #makes everything red 

    tex_preds = np.expand_dims(tex_preds, axis=0)

    print(np.unique(tex_preds))

    #create and visualize test preds 

    #save_preds expects: geom.shape(5, 64, 64, 64), tex_preds.shape(X, 2, 64, 64, 64)

    visualize.save_preds(geom[0], tex_preds, "Epoch_0_Step_0_TestTexs.png", "pointcloud", True, tex_targs)

    return geom, tex_preds, tex_targs

#Load VoxelDataset 
#get tex targs 
#print output to command line to visualize a single example 
def print_targs(geom, tex):

    z, y, x = np.nonzero(geom[0])  # see which voxels are occupied
    c = tex[0, z, y, x]

    xyz = np.vstack((x, y, z)).T

    utils.show_pointcloud(xyz, c)

    print(xyz, c)

    return geom, c


def check_loss(geom, tex_preds, tex_targs):
    #create a prediction with half and half blue and red 
    #count the number of red and blue 

    tex_preds = np.expand_dims(tex_preds, axis=0)

    train_loss_fn = TextureLoss() #Cross entropy 
    val_loss_fn = DiverseLoss(is_train=False, eval_mode=True) #Classification error

    #convert to tensors 
    tex_targs = tex_targs.astype(np.int64)
    tex_preds = torch.from_numpy(tex_preds)
    tex_targs = torch.from_numpy(tex_targs)

    #loss expects:
        # Preds:  torch.Size([1, 1, 2, 64, 64, 64])
        # Targs:  torch.Size([1, 1, 64, 64, 64])

    print(tex_preds.dtype, tex_targs.dtype)

    preds = tex_preds.view(*tex_preds.shape[:3], -1)
    targs = tex_targs.view(*tex_targs.shape[:2], -1)
    N, Ep, _, P = preds.shape
    _, Et, _ = targs.shape

    train_preds = preds[0].repeat(Et, 1, 1)
    train_targs = targs[0].repeat(1, Ep).view(-1, P) 

    train_loss = train_loss_fn(train_preds, train_targs)
    val_loss = val_loss_fn(tex_preds, tex_targs)

    print("Train Loss", train_loss, "Val Loss", val_loss)

    return val_loss


#check whether the classification loss and cross entropy are accurate. 

if __name__ == '__main__':
    dset = VoxelDataset(osp.join('data', 'voxelized_meshes'), 'use', is_train=True, random_rotation=0, n_ensemble=1)
    # print_targs(dset)
    geoms, preds, targs = create_test_meshes(dset)
    loss = check_loss(geoms, preds, targs)

    print(geoms.shape, targs.shape)
