"""
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3.7 save_obj_single.py --pose_path "/home/xiaoyu/tm_cnn/data/amass/data/BABEL/3d_obj/raw/KIT/917/Experiment3a_04_poses.npz" --class_name 6 --output_name 1b00442c-ff1a-4675-8d73-21e680545819 --start_ts 110.382 --end_ts 112.167
"""


import torch
import numpy as np
import sys
import argparse
sys.path.append('/home/xiaoyu/src/amass/human_body_prior/src/')

parser = argparse.ArgumentParser()
parser.add_argument("--pose_path",
                    type=str,
                    required=True,
                    help="The pose file path")
parser.add_argument("--class_name",
                    type=str,
                    required=True,
                    help="class name")
parser.add_argument("--output_name",
                    type=str,
                    required=True,
                    help="output video name")
parser.add_argument("--start_ts",
                    type=float,
                    required=False,
                    help="action start timestamp")
parser.add_argument("--end_ts",
                    type=float,
                    required=False,
                    help="action end timestamp")


args = parser.parse_args()


from human_body_prior.tools.omni_tools import copy2cpu as c2c
from os import path as osp

support_dir = '/home/xiaoyu/src/amass/support_data/'

# Choose the device to run the body model on.
comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# amass_npz_fname = osp.join(support_dir, 'github_data/amass_sample.npz') # the path to body data
# amass_npz_fname = osp.join(support_dir, 'github_data/01_01_poses.npz') # the path to body data
amass_npz_fname = args.pose_path
bdata = np.load(amass_npz_fname)

# you can set the gender manually and if it differs from data's then contact or interpenetration issues might happen
subject_gender = bdata['gender']
mocap_framerate = bdata['mocap_framerate']

if mocap_framerate < 25:
    frame_step = 1
else:
    frame_step = mocap_framerate // 25

print('Data keys available:%s'%list(bdata.keys()))

print('The subject of the mocap sequence is  {}.'.format(subject_gender))

from human_body_prior.body_model.body_model import BodyModel

bm_fname = osp.join(support_dir, 'body_models/smplh/{}/model.npz'.format(subject_gender))
dmpl_fname = osp.join(support_dir, 'body_models/dmpls/{}/model.npz'.format(subject_gender))

num_betas = 16 # number of body parameters
num_dmpls = 8 # number of DMPL parameters

bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(comp_device)
faces = c2c(bm.f)

time_length = len(bdata['trans'])

import random
# bdata['poses'][:, 3:66] = bdata['poses'][:, 3:66] * (-1)
for i in range(63):
    rand_factor = random.uniform(0, 1)
    bdata['poses'][:, i] = bdata['poses'][:, i] * rand_factor


if args.start_ts:
    start_frame = int(args.start_ts * mocap_framerate)
    end_frame = int(args.end_ts * mocap_framerate)
else:
    start_frame = 0
    end_frame = time_length

body_parms = {
    'root_orient': torch.Tensor(bdata['poses'][start_frame:end_frame+1, :3]).to(comp_device), # controls the global root orientation
    'pose_body': torch.Tensor(bdata['poses'][start_frame:end_frame+1, 3:66]).to(comp_device), # controls the body
    'pose_hand': torch.Tensor(bdata['poses'][start_frame:end_frame+1, 66:]).to(comp_device), # controls the finger articulation
    'trans': torch.Tensor(bdata['trans'][start_frame:end_frame+1]).to(comp_device), # controls the global body position
    'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=int(bdata['poses'][start_frame:end_frame+1, :3].shape[0]), axis=0)).to(comp_device), # controls the body shape. Body shape is static
    'dmpls': torch.Tensor(bdata['dmpls'][start_frame:end_frame+1, :num_dmpls]).to(comp_device) # controls soft tissue dynamics
}



print('root_orient = ', body_parms['root_orient'])

print('Body parameter vector shapes: \n{}'.format(' \n'.join(['{}: {}'.format(k,v.shape) for k,v in body_parms.items()])))
print('time_length = {}'.format(time_length))



import trimesh
from body_visualizer.tools.vis_tools import colors
# from body_visualizer.mesh.mesh_viewer import MeshViewer
from mesh_viewer_ground import MeshViewer
from body_visualizer.mesh.sphere import points_to_spheres
# from body_visualizer.tools.vis_tools import show_image

imw, imh=1600, 1600

body_trans_root = bm(**{k:v for k,v in body_parms.items() if k in ['pose_body', 'betas', 'pose_hand', 'dmpls',
                                                                   'trans','root_orient']})

def show_image(img_ndarray, img_name):
    '''
    Visualize rendered body images in Jupyter notebook
    :param img_ndarray: Nxim_hxim_wx3
    '''
    import matplotlib.pyplot as plt
    import cv2
    fig = plt.figure(figsize=(4, 4), dpi=300)
    ax = fig.gca()

    img = img_ndarray.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('{}'.format(img_name), img)
    plt.close(fig)


import tensorflow as tf
def vis_body_transformed_single(out_path, fId = 0):
    body_mesh = trimesh.Trimesh(vertices=c2c(body_trans_root.v[fId]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
    with tf.io.gfile.GFile(out_path, "w") as fout:
        body_mesh.export(fout, file_type="obj")



from pathlib import Path
import os
def vis_body_45():

    print('start_frame, end_frame = ', start_frame, end_frame)
    for t in range(start_frame, end_frame-1, int(frame_step)):
        # vis_body_transformed_single()
        out_dir_3 = '/home/xiaoyu/tm_cnn/data/amass/data/BABEL/3d_obj/kit_obj/{}/{}'.format(
            args.class_name,
            args.output_name)
        os.makedirs(out_dir_3, exist_ok=True)

        vis_body_transformed_single('/home/xiaoyu/tm_cnn/data/amass/data/BABEL/3d_obj/kit_obj/{}/{}/{}.obj'.format(
            args.class_name,
            args.output_name,str(t+start_frame)), fId=t-start_frame)


vis_body_45()
