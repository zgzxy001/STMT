import os
import tqdm
import imageio
import numpy as np
import time
import random
import math
import scipy.io as sio
from PIL import Image  #
import glob
import scipy
import argparse

"""
Prepare the dataloader format. 
Please modify Line 72 and Line 73 to your data path.  
"""

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--start_num', default=0,
                    help='start line number')
parser.add_argument('--end_num', default=0,
                    help='end line number')

args = parser.parse_args()

start_line = args.start_num
end_line = args.end_num


fx = 260.0 - 20
fy = 240
cx = (20.0 + 260) / 2
cy = (0 + 240) / 2

SAMPLE_NUM = 2048
fps_sample_num = 512
K = 24  # max frame limit for temporal rank
sample_num_level1 = 512

# save_path = 'C:\\Users\\Administrator\\Desktop\\LX\\paper\\dataset\\Prosessed_dataset\\01_MSR3D\\T'
save_path = './kit_pt_geodist_Processed_dataset'
try:
    os.makedirs(save_path)
except OSError:
    pass


def off_to_xyz(off_path):
    with open(off_path, 'r') as off_file:
        off_file.readline()
        off_file.readline()

        x_lst = []
        y_lst = []
        z_lst = []
        for line in off_file:
            # print(line)
            line_spt = line.strip().split(' ')
            if len(line_spt) == 4:
                break
            x, y, z = line_spt
            x_lst.append(x)
            y_lst.append(y)
            z_lst.append(z)
        points3d = np.array([x_lst, y_lst, z_lst])
        return points3d
        # return

def main():
    data_path = './off_path' # replace it to the folder that stores all the .off files produced by 02_obj2off.py
    geodesic_path = './off_path_geodist_path' # replace it to the folder that stores all the .npy files produced by 03_compute_geodist.py
    for class_dir in glob.iglob(data_path+'/*'):
        class_name = class_dir.split('/')[-1]
        for video_dir in glob.iglob(class_dir+'/*'):
            # video_name = video_dir.ss
            filename = video_dir.split('/')[-1]
            print(filename)
            geo_filename = filename+'_geodist'
            file = os.path.join(save_path, filename+'.npy')

            if os.path.isfile(file):
                continue

            off_name_lst = list(os.listdir(video_dir))
            off_name_lst.sort(key=lambda x:int(x.split('.')[0]))

            geo_name_lst = list(os.listdir(geodesic_path+'/'+class_name+'/'+filename))
            geo_name_lst.sort(key=lambda x:int(x.split('.')[0]))


            n_frame = len(off_name_lst)
            all_sam = np.arange(n_frame)
            if n_frame > K:
                frame_index = []
                for jj in range(K):
                    iii = int((int(n_frame * jj / K) + int(
                        n_frame * (jj + 1) / K)) / 2)
                    frame_index.append(iii)
                n_frame = K
            else:
                frame_index = [random.choice(all_sam) for _ in range(K-n_frame)]
                frame_index.extend(list(all_sam))
                n_frame = K
            frame_index.sort()
            all_frame_points_list = []
            all_frame_geo_list = []

            for idx in frame_index:

                curr_mesh_path = video_dir + '/'+ off_name_lst[idx]
                cloud_im = off_to_xyz(curr_mesh_path)
                all_frame_points_list.append(
                    cloud_im)

                curr_geo_path = geodesic_path+'/'+class_name+'/'+filename +'/' + geo_name_lst[idx]
                geo_data = scipy.io.loadmat(curr_geo_path)['geod_dist']
                all_frame_geo_list.append(geo_data)
              


            all_frame_3Dpoints_array = np.zeros(
                shape=[n_frame, SAMPLE_NUM, 3])
            all_geo_data = np.zeros(shape=[n_frame, SAMPLE_NUM,SAMPLE_NUM])
            for i in range(n_frame):
                curr_geo_data = all_frame_geo_list[i]
                each_frame_points = all_frame_points_list[i].T  # n*3

                if len(each_frame_points) < SAMPLE_NUM:  # lx#
                    if len(each_frame_points) < SAMPLE_NUM / 2:
                        if len(each_frame_points) < SAMPLE_NUM / 4:

                            rand_points_index = np.random.randint(0,
                                                                  each_frame_points.shape[
                                                                      0],
                                                                  size=SAMPLE_NUM - len(
                                                                      each_frame_points) - len(
                                                                      each_frame_points) - len(
                                                                      each_frame_points) - len(
                                                                      each_frame_points))

                            each_frame_points = np.concatenate((
                                each_frame_points,
                                each_frame_points,
                                each_frame_points,
                                each_frame_points,
                                each_frame_points[
                                rand_points_index,
                                :]), axis=0)

                        else:

                            rand_points_index = np.random.randint(0,
                                                                  each_frame_points.shape[
                                                                      0],
                                                                  size=SAMPLE_NUM - len(
                                                                      each_frame_points) - len(
                                                                      each_frame_points))
                            each_frame_points = np.concatenate((
                                each_frame_points,
                                each_frame_points,
                                each_frame_points[
                                rand_points_index,
                                :]), axis=0)

                    else:
                        rand_points_index = np.random.randint(0,
                                                              each_frame_points.shape[
                                                                  0],
                                                              size=SAMPLE_NUM - len(
                                                                  each_frame_points))
                        each_frame_points = np.concatenate((
                            each_frame_points,
                            each_frame_points[
                            rand_points_index,
                            :]), axis=0)
                        
                else:  
                    rand_points_index = np.random.randint(0,
                                                          each_frame_points.shape[
                                                              0],
                                                          size=SAMPLE_NUM)
                    each_frame_points = each_frame_points[
                                        rand_points_index, :]
                    curr_geo_data = curr_geo_data[rand_points_index, :]
                    curr_geo_data = curr_geo_data[:, rand_points_index]
                    
                each_frame_points = each_frame_points.astype(np.float)
                curr_geo_data = curr_geo_data.astype(np.float32)
                
            
                sampled_idx_l1 = farthest_point_sampling_fast(
                    each_frame_points, sample_num_level1)
                
                other_idx = np.setdiff1d(np.arange(SAMPLE_NUM),
                                            sampled_idx_l1.ravel())
                
                new_idx = np.concatenate(
                    (sampled_idx_l1.ravel(), other_idx))
                
                each_frame_points = each_frame_points[new_idx, :]
                
                curr_geo_data = curr_geo_data[new_idx, :]
                curr_geo_data = curr_geo_data[:, new_idx]


                
                all_frame_3Dpoints_array[
                    i] = each_frame_points  
                all_geo_data[i] = curr_geo_data
                
            save_npy(all_frame_3Dpoints_array, filename)
          
            all_geo_data_slice = all_geo_data[:, :sample_num_level1, :sample_num_level1]
            np.savez(os.path.join(save_path, geo_filename+'.npz'), all_geo_data_slice)
            
def save_npy(data, filename):
    file = os.path.join(save_path, filename)
    if not os.path.isfile(file):
        np.save(file, data)


def farthest_point_sampling_fast(pc, sample_num):
    pc_num = pc.shape[0]

    sample_idx = np.zeros(shape=[sample_num, 1], dtype=np.int32)
    sample_idx[0] = np.random.randint(0, pc_num)

    cur_sample = np.tile(pc[sample_idx[0], :], (pc_num, 1))

    
    diff = pc - cur_sample


    min_dist = (diff * diff).sum(axis=1)  #

    for cur_sample_idx in range(1, sample_num):
        

        sample_idx[cur_sample_idx] = np.argmax(min_dist)
        
        if cur_sample_idx < sample_num - 1:
            diff = pc - np.tile(pc[sample_idx[cur_sample_idx], :], (pc_num, 1))
            min_dist = np.concatenate((min_dist.reshape(pc_num, 1),
                                       (diff * diff).sum(axis=1).reshape(
                                           pc_num, 1)), axis=1).min(
                axis=1)  
    return sample_idx  


if __name__ == '__main__':
    main()

