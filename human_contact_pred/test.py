from inspect import stack
import numpy as np
from torch import float64
from create_animation import animate
import open3d
import os
import pickle
from create_animation import animate
import matplotlib.pyplot as plt
import transforms3d.euler as txe

osp = os.path

def show_predictions(dir, selections, stack=True):
  #pkl files saved as 
        # 'epoch': engine.state.epoch,
        # 'loss': loss, 
        # 'geom': geom,
        # 'tex_preds': tex_preds,
        # 'tex_targs': tex_targs
  #dir - directory of saved predictions
  #selections - list with indexes of desired files
   
  files = [f for f in os.listdir(dir) if osp.isfile(osp.join(dir, f))]

  for idx in selections:
    print("Epoch", idx, "prediction")
    filename = osp.expanduser(osp.join(dir, files[idx]))

    print(filename)
    
    geoms = []
    tex_preds = []

    if stack:
      with open(filename, 'rb') as f:
        d = pickle.load(f)

      print(len(d['geom'][0]))
      geom, _ = stack_textures(d, from_model=True)

      geoms.append(geom)
      # tex_preds.append(tex_pred)

    else: 
      #iterate through 3 predictions and visualize each one 
      for i in [0, 4, 9]:
        #visualize 
        print(i)
    
    for i in len(geoms):
      open3d.visualization.draw_geometries([geoms[i]])


def stack_textures(data, binary=False, from_model=False):
  #given a list of textures, add them all together and return a compoosite texture
  #assume that red is 1, and blue is 0, then the operation is logical or. 
  
  if (from_model):  
    cmap = np.asarray([[0, 0, 1], [1, 0, 0]])
    z, y, x = np.nonzero(data['geom'][0])
    pts = np.vstack((x, y, z)).T
    pred_idxs = range(1, len(data['tex_preds']))
    
    #initialize first point cloud 
    global_tex_pred = np.argmax(data['tex_preds'][0], axis=0)
    global_tex_pred = global_tex_pred[z, y, x]
    global_tex_pred = cmap[global_tex_pred]
    
    
    for pred_idx in pred_idxs:
        
        #load prediction 
        tex_pred = np.argmax(data['tex_preds'][pred_idx], axis=0)
        tex_pred = tex_pred[z, y, x]
        tex_pred = cmap[tex_pred]
        
        #merge loaded pointcloud into global pointcloud
        
        if binary: 
          global_tex_pred[:, 0] = np.maximum(global_tex_pred[:,0], tex_pred[:,0])
          global_tex_pred[:, 2] = np.minimum(global_tex_pred[:,2], tex_pred[:,2])
        else: 
          global_tex_pred[:, 2] += tex_pred[:,2]
          global_tex_pred[:, 0] += tex_pred[:,0]
        
    #render in open3d  
    global_tex_pred = global_tex_pred.astype(float)
    global_tex_pred[:,0] = global_tex_pred[:,0] / len(pred_idxs)
    global_tex_pred[:,2] = global_tex_pred[:,2] / len(pred_idxs)
  
    # print(np.unique(global_tex_pred, axis=0))
          
    g_geom = open3d.geometry.PointCloud()
    g_geom.points = open3d.utility.Vector3dVector(pts)
    g_geom.colors = open3d.utility.Vector3dVector(global_tex_pred)
    open3d.visualization.draw_geometries([g_geom])
    
    return (g_geom, global_tex_pred)

  else:
    #stack textures from ply mesh file
    
    print('Not Implemented')

def animate_stacks():
  filename = 'data/contactdb_predictions/mug_use_voxnet_diversenet_preds.pkl'
  filename = osp.expanduser(filename)
  with open(filename, 'rb') as f:
      d = pickle.load(f)
  geom, pred = stack_textures(d, binary=False, from_model=True)
  animate(geom)
    
if __name__ == '__main__':
#   dir = "assets/"
#   show_predictions(dir, [0], stack=True)
    animate_stacks()