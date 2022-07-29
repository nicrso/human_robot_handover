import utils
import os
import sys
import time 
import glob
import pathlib
import argparse
from itertools import chain
from pickle import TRUE

import numpy as np
from skimage.io import imsave
import cv2

import tensorflow,compat.v1 as tf 
tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from env import UR5PickEnviornment
from common import get_splits
from dataset import VoxelDataset, show_voxel_texture
osp = os.path

#figure out how to import files from path: "../../contact_graspnet/contact_graspnet"
sys.path.insert(0, "~/Desktop/Robotics/contact_graspnet/contact_graspnet")
import config_utils
from data import regularize_pc_point_count, depth2pc, load_available_input_data

from contact_grasp_estimator import GraspEstimator
from visualization_utils import visualize_grasps, show_image

def run_sim(task, type, global_config, checkpoint_dir, input_paths, K=None, local_regions=True, skip_border_objects=False, filter_grasps=True, segmap_id=None, z_range=[0.2,1.8], forward_passes=1):
    """
    Run simulation with contact_graspnet used to predict grasps 
    
    :param global_config: config.yaml from checkpoint directory
    :param checkpoint_dir: checkpoint directory
    :param input_paths: .png/.npz/.npy file paths that contain depth/pointcloud and optionally intrinsics/segmentation/rgb
    :param K: Camera Matrix with intrinsics to convert depth to point cloud
    :param local_regions: Crop 3D local regions around given segments. 
    :param skip_border_objects: When extracting local_regions, ignore segments at depth map boundary.
    :param filter_grasps: Filter and assign grasp contacts according to segmap.
    :param segmap_id: only return grasps from specified segmap_id.
    :param z_range: crop point cloud at a minimum/maximum z distance from camera to filter out outlier points. Default: [0.2, 1.8] m
    :param forward_passes: Number of forward passes to run on each point cloud. Default: 1
    """

    #initialize dataset
    grid_size = 64
    
    data_dir = osp.join('data', 'voxelized_meshes')
    
    kwargs = dict(data_dir=data_dir, instruction=type, train=True, random_rotation=0, n_ensemble=-1, test_only=False)
    
    dset = VoxelDataset(grid_size=grid_size, **kwargs)
    dset_names = list(dset.filenames.keys())
    
    # load model
    grasp_estimator = GraspEstimator(global_config)
    grasp_estimator.build_network()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver(save_relative_paths=True)

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    # Load weights
    grasp_estimator.load_weights(sess, saver, checkpoint_dir, mode='test')

    os.makedirs('results', exist_ok=True)

    # load env
    env = UR5PickEnviornment(gui=True)
    
    if task == 'pick_training':
         
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

                #Predict 6DOF grasp 

                #Required: segmap, rgb, depth, cam_K, pc_full, pc_colors 
                segmap, rgb, depth, cam_K, pc_full, pc_colors = None, None, None, None, None, None

                segmap=mask_obs

                rgb=rgb_obs
                rgb = np.array(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))

                depth=depth_obs

                cam_K = env.camera.intrinsic_matrix.reshape(3,3)
        
                if segmap is None and (local_regions or filter_grasps):
                    raise ValueError('Need segmentation map to extract local regions or filter grasps')

                if pc_full is None:
                    print('Converting depth to point cloud(s)...')
                    pc_full, pc_segments, pc_colors = grasp_estimator.extract_point_clouds(depth, cam_K, segmap=segmap, rgb=rgb,
                                                                                            skip_border_objects=skip_border_objects, z_range=z_range)

                print('Generating Grasps...')
                pred_grasps_cam, scores, contact_pts, _ = grasp_estimator.predict_scene_grasps(sess, pc_full, pc_segments=pc_segments, 
                                                                                                local_regions=local_regions, filter_grasps=filter_grasps, forward_passes=forward_passes)  

                # Save results
                np.savez('results/predictions_{}'.format(os.path.basename(p.replace('png','npz').replace('npy','npz'))), 
                        pred_grasps_cam=pred_grasps_cam, scores=scores, contact_pts=contact_pts)

                # Visualize results          
                show_image(rgb, segmap)
                visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors)
                

                #Execute grasp 
                # coord, angle, vis_img = model.predict_grasp(rgb_obs)
                # pick_pose = env.image_pose_to_pick_pose(coord, angle, depth_obs)
                # result = env.execute_grasp(*pick_pose)
                
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

    parser = argparse.ArgumentParser(description='Model eval script')
    parser.add_argument('-t', '--task', default='pick_training',
        help='which task to do: "pick_training" or "empty_bin"')
    parser.add_argument('--type', default='use',
        help='which object type to load: ycb, handoff, or use')
    parser.add_argument('--ckpt_dir', default='checkpoints/scene_test_2048_bs3_hor_sigma_001', 
        help='Log dir [default: checkpoints/scene_test_2048_bs3_hor_sigma_001]')
    parser.add_argument('--K', default=None, 
        help='Flat Camera Matrix, pass as "[fx, 0, cx, 0, fy, cy, 0, 0 ,1]"')
    parser.add_argument('--z_range', default=[0.2,1.8], 
        help='Z value threshold to crop the input point cloud')
    parser.add_argument('--local_regions', action='store_true', default=False, 
        help='Crop 3D local regions around given segments.')
    parser.add_argument('--filter_grasps', action='store_true', default=False,  
        help='Filter grasp contacts according to segmap.')
    parser.add_argument('--skip_border_objects', action='store_true', default=False,  
        help='When extracting local_regions, ignore segments at depth map boundary.')
    parser.add_argument('--forward_passes', type=int, default=1,  
        help='Run multiple parallel forward passes to mesh_utils more potential contact points.')
    #in pybullet object_id=5
    parser.add_argument('--segmap_id', type=int, default=0,  
        help='Only return grasps of the given object id')

    FLAGS = parser.parse_args()

    global_config = config_utils.load_config(FLAGS.ckpt_dir, batch_size=FLAGS.forward_passes, arg_configs=FLAGS.arg_configs)

    run_sim(FLAGS.task, FLAGS.type, global_config, FLAGS.ckpt_dir, FLAGS.np_path if not FLAGS.png_path else FLAGS.png_path, z_range=eval(str(FLAGS.z_range)),
                K=FLAGS.K, local_regions=FLAGS.local_regions, filter_grasps=FLAGS.filter_grasps, segmap_id=FLAGS.segmap_id, 
                forward_passes=FLAGS.forward_passes, skip_border_objects=FLAGS.skip_border_objects)
