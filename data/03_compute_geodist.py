# stdlib
from time import time
# 3p
import numpy as np
import scipy.io as sio
from tridesic import get_heat_geodesics, get_fmm_geodesics, get_exact_geodesics

import os
import glob
import numpy as np

import argparse

"""
Compute the geodist distance which is required for the computation of surface field convolution. 
The input is the .off file generated from 02_obj2off.py, the output should be saved to .npy format.
"""
parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--input_path', default=0,
                    help='start line number')
parser.add_argument('--out_path', default=0,
                    help='end line number')

args = parser.parse_args()

def read_off(file):
    file = open(file, "r")
    if file.readline().strip() != "OFF":
        raise "Not a valid OFF header"

    n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(" ")])
    verts = [[float(s) for s in file.readline().strip().split(" ")] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(" ")][1:] for i_face in range(n_faces)]

    return np.array(verts), np.array(faces)


def single_main(off_path, out_path):

    verts, faces = read_off(off_path)


    geodesic_dist = get_fmm_geodesics(verts, faces)
    geodesic_dist = np.float32(geodesic_dist)

    sio.savemat(out_path, {"geod_dist": geodesic_dist})


if __name__ == "__main__":
    off_path = args.input_path
    out_path = args.out_path
    single_main(off_path, out_path)
