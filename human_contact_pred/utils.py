import visualize
import numpy as np
import open3d
import os
import pickle
from torch import float64
import torchvision

osp = os.path

#update show_preds 

def reshape_data_for_logging(geom, tex_preds, tex_targs, train=True):

  if train: 
    if geom.shape[0] > 1:
      geom = geom[0]
      geom = geom.detach().cpu().numpy().squeeze()

    if tex_preds.shape[0] > 1:
        tex_preds = tex_preds[0]

    tex_preds = tex_preds.detach().cpu().numpy().squeeze()
    tex_targs = tex_targs.detach().cpu().numpy()

    if geom[0].shape[0] == 5:
        geom = geom[0]

    if tex_preds.shape[0] == 2:
        tex_preds = np.expand_dims(tex_preds, 0)
      
    return geom, tex_preds, tex_targs

  else: 
    geom = geom.detach().cpu().numpy()
    tex_preds = tex_preds.detach().cpu().numpy()
    tex_targs = tex_targs.detach().cpu().numpy()

    return geom, tex_preds, tex_targs

def load_from_pkl(data):
  """
    Load geoms and predictions from pickle. 

    :param data: Data returned from pick.load()
  """ 
  geom = data['geom'][0]
  preds = data['tex_preds']

  return geom, preds

def stack_textures(geom, preds, prediction=False, binary=False, show=False):
  """
  Given a list of textures, add them all together and return a compoosite texture
  assume that red is 1, and blue is 0, then the operation is logical or.

  :param geom: 
  :param preds: 
  :param binary: Defines whether to combine preds using a binary color scheme 
  :param show:  
  """

  print(preds.shape)

  #account for stacking from saved prediction 
  #account for stacking from model output 
 
  cmap = np.asarray([[0, 0, 1], [1, 0, 0]])

  z, y, x = np.nonzero(geom)
  pts = np.vstack((x, y, z)).T
  
  #initialize first point cloud 
  
  #change processing for predictions 
  if prediction:
    global_tex_pred = np.argmax(preds[0], axis=0)
  else: 
    global_tex_pred = preds[0]

  global_tex_pred = global_tex_pred[z, y, x]

  print(global_tex_pred.shape)

  global_tex_pred = cmap[global_tex_pred]

  pred_idxs = range(1, len(preds))
  
  for pred_idx in pred_idxs:
      
      #load prediction 
      tex_pred = np.argmax(preds[pred_idx], axis=0)
      tex_pred = tex_pred[z, y, x]
      tex_pred = cmap[tex_pred]
      
      #merge loaded pointcloud into global pointcloud

      if binary: 
        global_tex_pred[:, 0] = np.maximum(global_tex_pred[:,0], tex_pred[:,0])
        global_tex_pred[:, 2] = np.minimum(global_tex_pred[:,2], tex_pred[:,2])
      else: 
        global_tex_pred[:, 2] += tex_pred[:,2]
        global_tex_pred[:, 0] += tex_pred[:,0]
        
  global_tex_pred = global_tex_pred.astype(float)
  global_tex_pred[:,0] = global_tex_pred[:,0] / len(pred_idxs)
  global_tex_pred[:,2] = global_tex_pred[:,2] / len(pred_idxs)

  #render in open3d 
  if show:
    g_geom = open3d.geometry.PointCloud()
    g_geom.points = open3d.utility.Vector3dVector(pts)
    g_geom.colors = open3d.utility.Vector3dVector(global_tex_pred)
    open3d.visualization.draw_geometries([g_geom])
  
  return (geom, global_tex_pred)

def show_predictions(dir, selections, stack=True) -> None:
  """
    Render predictions from pickle file format in open3d.
    
    Where pkl files saved as: 
      'epoch': engine.state.epoch,
      'loss': loss, 
      'geom': geom,
      'tex_preds': tex_preds,
      'tex_targs': tex_targs
    
    :param dir: directory of saved predictions
    :param selections: list with indexes of desired files
    :param stack: Defines whether to stack predictions 
  """
   
  files = [f for f in os.listdir(dir) if osp.isfile(osp.join(dir, f))]

  for idx in selections:
    print("Epoch", idx, "prediction")
    filename = osp.expanduser(osp.join(dir, files[idx]))

    print(filename)
    
    geoms = []

    if stack:
      with open(filename, 'rb') as f:
        d = pickle.load(f)

      geom, preds = load_from_pkl(d)
      stacked_geom, _ = stack_textures(geom, preds, prediction=True)

      geoms.append(stacked_geom)

    else: 
      #iterate through 3 predictions and visualize each one 
      for i in [0, 4, 9]:
        #visualize 
        print(i)
    
    for i in len(geoms):
      open3d.visualization.draw_geometries([geoms[i]])

def texture_proc(colors, a=0.05, invert=False):

  idx = colors > 0
  ci = colors[idx]
  if len(ci) == 0:
    return colors
  if invert:
    ci = 1 - ci
  # fit a sigmoid
  x1 = min(ci); y1 = a
  x2 = max(ci); y2 = 1-a
  lna = np.log((1 - y1) / y1)
  lnb = np.log((1 - y2) / y2)
  k = (lnb - lna) / (x1 - x2)
  mu = (x2*lna - x1*lnb) / (lna - lnb)
  # apply the sigmoid
  ci = np.exp(k * (ci-mu)) / (1 + np.exp(k * (ci-mu)))
  colors[idx] = ci
  return colors


def discretize_texture(c, thresh=0.4, have_dontcare=True):

  idx = c > 0
  if sum(idx) == 0:
    return c
  ci = c[idx]
  c[:] = 2 if have_dontcare else 0
  ci = ci > thresh
  c[idx] = ci
  return c

def show_pointcloud(pts, colors, cmap=np.asarray([[0,0,1],[1,0,0],[0,0,1]])):

  colors = np.asarray(colors)
  if (colors.dtype == int) and (colors.ndim == 1) and (cmap is not None):
    colors = cmap[colors]
  if colors.ndim == 1:
    colors = np.tile(colors, (3, 1)).T

  pc = open3d.geometry.PointCloud()
  pc.points = open3d.utility.Vector3dVector(np.asarray(pts))
  pc.colors = open3d.utility.Vector3dVector(colors)

  open3d.visualization.draw_geometries([pc])

def combine_meshes():
  srcdir = osp.join("data", "voxelized_meshes")
  outdir = osp.join("data", "voxelized_meshes_combined")

  #parse each file individually, keep a running dictionary of the filename, count, geom, and preds

  files = os.listdir(srcdir)
  
  file_dict = {}

  for filename in files: 
    object_name = filename.split('_')
    object_name = object_name[1] + '_' + object_name[2]

    if "_solid.npy" not in filename:
      continue

    if "testonly" in filename:
      continue

    if object_name not in file_dict.keys():
      file_dict[object_name] = [filename]
       
    else: 
      file_dict[object_name].append(filename)

  #loop through each object in file_dict 
  for object in file_dict.keys():
    
    #load in points 
    x, y, z, c, xx, yy, zz = np.load(osp.join(srcdir, file_dict[object][0]))
    x, y, z = x.astype(int), y.astype(int), z.astype(int)
    pts = np.vstack((xx, yy, zz))
    offset = (pts.max(1, keepdims=True) + pts.min(1, keepdims=True)) / 2
    pts -= offset
    scale = max(pts.max(1) - pts.min(1)) / 2
    pts /= scale
    pts = np.vstack((np.ones(pts.shape[1]), pts, scale*np.ones(pts.shape[1])))
    
    # center the object
    offset_x = (64 - x.max() - 1) // 2
    offset_y = (64 - y.max() - 1) // 2
    offset_z = (64 - z.max() - 1) // 2
    x += offset_x
    y += offset_y
    z += offset_z

    # create occupancy grid
    geom = np.zeros((5, 64, 64, 64),
      dtype=np.float32)
    geom[:, z, y, x] = pts

    texs = []

    file_lst = file_dict[object]

    for filename in file_lst:
      filename = osp.join(srcdir, filename)
      _, _, _, c, _, _, _ = np.load(filename)
      c = discretize_texture(c, thresh=0.4)
      tex = 2 * np.ones((64, 64, 64),
        dtype=np.float32)
      tex[z, y, x] = c
      texs.append(tex)

    texs = np.stack(texs)

    geom = geom.astype(np.float32)
    texs = texs.astype(np.int)

    #combine (stack) textures
    geom, pred = stack_textures(geom[0], texs, prediction=False)

    im_name = str(object) + ".png"

    #visualize 
    visualize.save_preds(geom, pred, im_name)

    #save to output directory as np file

  return 

if __name__ == "__main__":
  combine_meshes()