import utils
import os
import pathlib
import argparse
from itertools import chain
from pickle import TRUE

import torch
import numpy as np
from skimage.io import imsave

from env import UR5PickEnviornment
from common import load_chkpt, get_splits
import affordance_model
#import action_regression_model
from dataset import VoxelDataset, show_voxel_texture
osp = os.path

def main():
    parser = argparse.ArgumentParser(description='Model eval script')
    parser.add_argument('-m', '--model', default="affordance",
        help='which model to train: "affordance" or "action_regression"')
    parser.add_argument('-t', '--task', default='pick_training',
        help='which task to do: "pick_training" or "empty_bin"')
    parser.add_argument('--headless', action='store_true',
        help='launch pybullet GUI or not')
    parser.add_argument('--seed', type=int, default=10000000,
        help='random seed for empty_bin task')
    parser.add_argument('--type', default='use',
        help='which object type to load: ycb, handoff, or use')
    args = parser.parse_args()

    if args.model == 'action_regression':
        #model_class = action_regression_model.ActionRegressionModel
        print("Failed")
    else:
        model_class = affordance_model.AffordanceModel

    model_dir = os.path.join('data', args.model)
    chkpt_path = os.path.join(model_dir, 'best.ckpt')

    #initialize dataset
    grid_size = 64
    
    data_dir = osp.join('data', 'voxelized_meshes')
    
    kwargs = dict(data_dir=data_dir, instruction=args.type, train=True, random_rotation=0, n_ensemble=-1, test_only=False)
    
    dset = VoxelDataset(grid_size=grid_size, **kwargs)
    dset_names = list(dset.filenames.keys())
    
    # load model
    device = torch.device("cpu")
    print(device, chkpt_path)
    model = model_class()
    model.to(device)
    load_chkpt(model, chkpt_path, device)
    model.eval()
    
    #get object types 
    type = args.type

    # load env
    env = UR5PickEnviornment(gui=not args.headless)
    
    if args.task == 'pick_training':
        
        #edit 
        obj_names = get_splits(type)['train']
        
        obj_names = [i for i in obj_names if i[len(type)+1:] in dset_names]

        print(obj_names)
        
        n_attempts = 1
        vis_dir = os.path.join(model_dir, 'eval_pick_training_vis')
        pathlib.Path(vis_dir).mkdir(parents=True, exist_ok=True)

        results = list()
        for name_idx, name in enumerate(obj_names):
            print('Picking: {}'.format(name))
            env.remove_objects()
            for i in range(n_attempts):
                print('Attempt: {}'.format(i))
                seed = name_idx * 100 + i + 10000
                if i == 0:
                    env.load_objects([name], type, seed=seed)
                else:
                    env.reset_objects(seed)
                
                #get RGB-D data
                rgb_obs, depth_obs, mask_obs = env.observe()
                
                #save npy array of scene
                #save camera.intrinsic_matrix, rgb, depth, segmap

                # print(rgb_obs.shape, depth_obs.shape, env.camera.intrinsic_matrix.shape, mask_obs.shape)
                save_obj = {'rgb': rgb_obs, 'depth': depth_obs, 'K': env.camera.intrinsic_matrix, 'seg': mask_obs}
                print(mask_obs.shape, np.unique(mask_obs))
                np.save('{}.npy'.format(name), save_obj)

                #show object voxels
                idx = dset_names.index(name[len(type)+1:])

                pts = show_voxel_texture(dset[idx])
                
                coord, angle, vis_img = model.predict_grasp(rgb_obs)
                pick_pose = env.image_pose_to_pick_pose(coord, angle, depth_obs)
                result = env.execute_grasp(*pick_pose)
                
                #pass voxel map to model 
                #get voxels that can be picked up 
                #match to observed image 
                #execute pickup 
                
                print('Success!' if result else 'Failed:(')
                fname = os.path.join(vis_dir, '{}_{}.png'.format(name, i))
                imsave(fname, vis_img)
                results.append(result)
                
        success_rate = np.array(results, dtype=np.float32).mean()
        print("Success rate: {}".format(success_rate))
    else:
        print('other')


if __name__ == '__main__':
    main()
