'''
This script generates screenshots of the object rotating around the up vector
in the animation_images directory.
You can upload them to online services like https://ezgif.com/maker
to create animation GIFs
'''

import os
import open3d
import numpy as np
from utils import texture_proc
import transforms3d.euler as txe
from IPython.core.debugger import set_trace
import matplotlib.pyplot as plt
import pickle
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
osp = os.path

def animate(geom, n_images=42, save=True, suffix=None):
  """
  Visualizes n_images around an object from an inputted geometry with texture 
  """

  if save:
    images = []
  else : 
    images = None

  # adjust this transform as needed
  T = np.eye(4)
  T[:3, :3] = txe.euler2mat(np.deg2rad(-90), np.deg2rad(0), np.deg2rad(0))
  geom.transform(T)

  animate.count = -1
  animate.step = 50.0  # simulates mouse cursor movement by 50 pixels
  animate.radian_per_pixel = 0.003

  def move_forward(vis):
    glb = animate
    ctr = vis.get_view_control()
    ro = vis.get_render_option()
    ro.point_size = 25.0
    if glb.count >= 0:
      image = vis.capture_screen_float_buffer(False)
      
      if save:
        t_image = image.ToTensor()
        if t_image.is_tensor():
          print("T")
        images.append(t_image)	  

      im_filename = osp.join('animation_images',
          'image_{:03d}'.format(glb.count))
      if suffix is not None:
        im_filename = '{:s}_{:s}'.format(im_filename, str(suffix))
      im_filename += '.png'
      plt.imsave(im_filename, np.asarray(image))

      if np.rad2deg(glb.radian_per_pixel * glb.step * glb.count) >= 360.0:
        vis.register_animation_callback(None)

      ctr.rotate(glb.step, 0)
    else:
      # no effect, adjust as needed. Higher values cause the view to zoom out
      ctr.scale(1)  
    glb.count += 1

  open3d.visualization.draw_geometries_with_animation_callback([geom], move_forward)

  return images


if __name__ == '__main__':
  MESH = False
  # for a textured mesh model (e.g. ContactDB contactmap)
  if MESH:
    filename = 'data/contactdb_contactmaps/42_handoff_cylinder_large.ply'
    filename = osp.expanduser(filename)
    geom = open3d.io.read_triangle_mesh(filename)
    geom.compute_vertex_normals()
    geom.compute_triangle_normals()

    c = np.asarray(geom.vertex_colors)[:, 0]
    c = texture_proc(c)
    c = plt.cm.inferno(c)[:, :3]
    geom.vertex_colors = open3d.utility.Vector3dVector(c)
    animate(geom)
  else:  # for the voxelgrid and pointcloud predictions made by ContactDB models
    #filename = 'data/contactdb_predictions/camerav2_use_voxnet_diversenet_preds.pkl'
    filename = 'data/contactdb_predictions/mug_use_voxnet_diversenet_preds.pkl'
    filename = osp.expanduser(filename)
    with open(filename, 'rb') as f:
      d = pickle.load(f)
    cmap = np.asarray([[0, 0, 1], [1, 0, 0]])
    z, y, x = np.nonzero(d['geom'][0])
    pts = np.vstack((x, y, z)).T
    pred_idxs = [0, 4, 9]
    #pred_idxs = range(len(d['tex_preds']))
    for pred_idx in pred_idxs:
      print(pred_idx)
      tex_pred = np.argmax(d['tex_preds'][pred_idx], axis=0)
      tex_pred = tex_pred[z, y, x]
      tex_pred = cmap[tex_pred]
      geom = open3d.geometry.PointCloud()
      geom.points = open3d.utility.Vector3dVector(pts)
      geom.colors = open3d.utility.Vector3dVector(tex_pred)

      images = animate(geom, save=True)
      grid = make_grid(images)

      grid_filename = osp.join('animation_images',
          'grid_{:03d}'.format(pred_idx))
      plt.imsave(grid_filename, np.asarray(grid))
      #open3d.draw_geometries([geom])

