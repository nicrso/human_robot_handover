import numpy as np
import matplotlib
matplotlib.use('Agg')

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import io
import os
from PIL import Image
from skimage.measure import block_reduce
import matplotlib.cm as cm
import matplotlib as mpl
import voxel_dataset

osp = os.path

def pointcloud_to_voxels(xyz, c):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    filled = np.zeros((64, 64, 64), dtype=bool)
    filled[x, y, z] = True

    fc = np.empty(filled.shape + (3,), dtype=object)
    fc[x, y, z] = c

    #Possible additions: 
    #  1) post process to remove inside voxels
    #  2) avg pooling 

    return filled, fc

def set_view_and_save_img(fig, ax, views):
    for elev, azim in views:
        ax.view_init(elev=elev, azim=azim)
        yield plot_to_png(fig)

def plot_to_png(fig):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = np.array(
        Image.open(buf)).astype(np.uint8)
    return img

def view_tsdf(tsdf, simplify=False):
    main_color = '#00000055'
    mpl.rcParams['text.color'] = main_color
    mpl.rcParams['axes.labelcolor'] = main_color
    mpl.rcParams['xtick.color'] = main_color
    mpl.rcParams['ytick.color'] = main_color
    mpl.rc('axes', edgecolor=main_color)
    mpl.rcParams['grid.color'] = '#00000033'
    if simplify:
        tsdf = block_reduce(tsdf, block_size=(2, 2, 2), func=np.mean)
    x = np.arange(tsdf.shape[0])[:, None, None]
    y = np.arange(tsdf.shape[0])[None, :, None]
    z = np.arange(tsdf.shape[0])[None, None, :]
    x, y, z = np.broadcast_arrays(x, y, z)
    c = cm.plasma((tsdf.ravel() + 1))
    alphas = (tsdf.ravel() < 0).astype(float)
    c[..., -1] = alphas
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(x.ravel(),
               y.ravel(),
               z.ravel(),
               c=c,
               s=1)
    ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    # Hide axes ticks
    ax.tick_params(axis='x', colors=(0.0, 0.0, 0.0, 0.0))
    ax.tick_params(axis='y', colors=(0.0, 0.0, 0.0, 0.0))
    ax.tick_params(axis='z', colors=(0.0, 0.0, 0.0, 0.0))
    ax.view_init(20, -110)
    plt.show()

#features expects a Nx1 array or Nx3
def plot_3d(xyz, features, 
                    plot_type="pointcloud",
                    object_labels=None,
                    background_color=(0.1, 0.1, 0.1, 0.99),
                    num_points=20000, views=[(45, 135)],
                    pts_size=3, alpha=0.5, plot_empty=False,
                    visualize_ghost_points=False,
                    object_colors=None,
                    delete_fig=True,
                    show_plot=False):

    is_semantic = len(features.shape) == 1
    if type(alpha) is float:
        alpha = np.ones(xyz.shape[0]).astype(np.float32)*alpha
    if not plot_empty and is_semantic and\
            object_labels is not None and\
            len(list(filter(lambda l: 'empty' in l, object_labels.tolist()))) > 0:
        empty_idx = object_labels.tolist().index(
            list(filter(lambda l: 'empty' in l, object_labels.tolist()))[0])
        mask = features != empty_idx
        xyz = xyz[mask, :]
        features = features[mask, ...]
        alpha = alpha[mask]
        if type(pts_size) != int and type(pts_size) != float:
            pts_size = pts_size[mask]

    # subsample
    # if xyz.shape[0] > num_points:
    #     indices = np.random.choice(
    #         xyz.shape[0],
    #         size=num_points,
    #         replace=False)
    #     xyz = xyz[indices, :]
    #     features = features[indices, ...]
    #     alpha = alpha[indices]
    #     if type(pts_size) != int and type(pts_size) != float:
    #         pts_size = pts_size[indices]

    fig = plt.figure(figsize=(3, 3), dpi=60)
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    ax.set_facecolor(background_color)
    ax.w_xaxis.set_pane_color(background_color)
    ax.w_yaxis.set_pane_color(background_color)
    ax.w_zaxis.set_pane_color(background_color)
    # ax.set_axis_off()
    # ax._axis3don = False
    if is_semantic and object_labels is not None:
        object_ids = list(np.unique(features))
        object_labels = object_labels[object_ids].tolist()
        if object_colors is not None:
            object_colors = object_colors[object_ids]
        features = features.astype(np.int)
        # repack object ids
        repacked_obj_ids = np.zeros(features.shape).astype(np.uint32)
        for i, j in enumerate(object_ids):
            repacked_obj_ids[features == j] = i
        features = repacked_obj_ids
        object_ids = list(np.unique(features))
        colors = np.zeros((len(features), 4)).astype(np.uint8)
        if object_colors is None:
            cmap = plt.get_cmap('tab20')
            object_colors = (255*cmap(np.array(object_ids) %
                             20)).astype(np.uint8)
        for obj_id in np.unique(features):
            colors[features == obj_id, :] = object_colors[obj_id]
        colors = colors.astype(float)/255.
        object_colors = object_colors.astype(float)/255
        handles = [Patch(facecolor=c, edgecolor='grey',
                         label=label) for label, c in zip(object_labels, object_colors)]
        l = ax.legend(handles=handles,
                      labels=object_labels,
                      loc='lower center',
                      bbox_to_anchor=(0.5, 0),
                      ncol=4,
                      facecolor=(0, 0, 0, 0.1),
                      fontsize=8,
                      framealpha=0)
        plt.setp(l.get_texts(), color=(0.8, 0.8, 0.8))
    else:
        colors = features.astype(float)
        if colors.max() > 1.0:
            colors /= 255.
            assert colors.max() <= 1.0
    # ensure alpha has same dims as colors
    if colors.shape[-1] == 4:
        colors[:, -1] = alpha

    if plot_type=="pointcloud":
        ax.scatter(x, y, z, c=colors, s=pts_size)
    elif plot_type=="voxel":
        #convert list of voxels to 64x64x64 3d array of values indicating voxels to fill
        filled, fc = pointcloud_to_voxels(xyz, colors)
        ax.voxels(filled, facecolors=fc)

    if visualize_ghost_points:
        x, y, z = np.array(np.unique(xyz, axis=0)).T
        ax.scatter(x, y, z, color=[1., 1., 1., 0.1], s=pts_size)
    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.axes.set_xlim3d(left=-0.0, right=64.0)
    ax.axes.set_ylim3d(bottom=-0.0, top=64.0)
    ax.axes.set_zlim3d(bottom=-0.0, top=64.0)
    plt.tight_layout(pad=0)
    imgs = list(set_view_and_save_img(fig, ax, views))
    if show_plot:
        plt.show()
    if delete_fig:
        plt.close(fig)
    return imgs

def process_voxels(geom, tex, is_pred=False):
    cmap=np.asarray([[0,0,1],[1,0,0],[0,0,1]]) # RGB
    z, y, x = np.nonzero(geom) #check which voxels are occupied

    if is_pred: 
        c = np.argmax(tex, axis=0)
        c = c[z, y, x]
    else: 
        c = tex[0, z, y, x] 

    c = cmap[c]   

    xyz = np.vstack((x,y,z)).T

    return xyz, c

def save_wandb_table():

    return 

def save_preds(geom, textures, filename, plot_type="pointcloud", is_pred=False, targs=None, model_type=None):
    """
        save_preds anticipates:
            geom.shape = [1, 64, 64, 64]
            textures.shape = [N, 2, 64, 64, 64]
            targs.shape = [N, 64, 64, 64]
    """


    vs = [(45, 45), (45, 135), (45, 225), (45, 315), (-45, 315), (-45, 225), (-45, 135), (-45, 45)]
    im_grid = []

    #1) add ground truths to grid 
    if targs is not None:
        xyz, c = process_voxels(geom, targs[0])
        row_imgs = plot_3d(xyz, c, plot_type=plot_type, views = vs)
        row_im_grid = np.concatenate(row_imgs, axis=1)
        im_grid.append(row_im_grid)
        print("Saved ground truth")

    #2) add stacking to save_preds

    #3) add stacked ground truths to im grid 


    for idx, tex in enumerate(textures):
        xyz, c = process_voxels(geom, tex, is_pred)

        row_imgs = plot_3d(xyz, c, plot_type=plot_type, views=vs)
        row_im_grid = np.concatenate(row_imgs, axis=1)
        im_grid.append(row_im_grid)
        print("Saved Img: " + str(idx))

    im_grid = np.vstack(im_grid)

    im = Image.fromarray(im_grid)

    #add diversenet or cnn3d to front of folder 

    target_folder = osp.join('data', 'training_images')

    dirs = [x for x in os.listdir(target_folder) if osp.isdir(osp.join(target_folder, x))]

    dirs_idxs = [int(x.split('_')[1]) for x in dirs]
    if len(dirs_idxs) == 0:
        dirs_idxs.append(-1)

    max_dir_index = max(dirs_idxs)

    # print("Max_Dir", max_dir_index)

    epoch = filename.split('_')[1]
    step = filename.split('_')[3]

    if int(epoch) == 0 and int(step)==0:
        new_dir = "version_" + str(max_dir_index+1) + "_" + model_type if model_type is not None else "version_" + str(max_dir_index+1)
        target_folder = osp.join(target_folder, new_dir)
        os.mkdir(target_folder)

    else:
        old_dir = "version_" + str(max_dir_index) + "_" + model_type if model_type is not None else "version_" + str(max_dir_index)
        target_folder = osp.join(target_folder, old_dir)

    #else, work in the folder with the highest index
    pred_filename = osp.join(target_folder, filename)
     
    print(pred_filename)
    im.save(pred_filename)

    return im

def fast_scandir(dirname):
    subfolders= [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(fast_scandir(dirname))
    return subfolders



if __name__ == "__main__":
    #load a pointcloud dataset

    n_ensemble = 1
    N_show = 1
    dset = voxel_dataset.VoxelDataset(osp.join('data', 'voxelized_meshes_test'), 'use',
        is_train=False, random_rotation=0, n_ensemble=n_ensemble)

    vs = [(45, 45), (45, 135), (45, 225), (45, 315), (-45, 315), (-45, 225), (-45, 135), (-45, 45)]

    for idx in range(N_show):
        geom, tex = dset[idx]
        xyz, c = process_voxels(geom[0], tex)
        imgs = plot_3d(xyz, c, views=vs)

        im_grid = np.concatenate(imgs, axis=1)
        im = Image.fromarray(im_grid)
        im.save("Voxel_Mug.png")
