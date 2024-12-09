import os
import glob
import numpy as np

import trimesh
import numpy as np
import argparse

"""
Convert .obj file format to .off file format, which is required for computing geodesic distance used in surface field convolution.
Need to download meshconv and put it in the ./meshlab/meshconv folder.
The input is .obj file generated using 01_blender_process.py. The output is saved to .off file format.
"""
parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--input_path', default=0,
                    help='start line number')
parser.add_argument('--out_path', default=0,
                    help='end line number')

args = parser.parse_args()

off_tmp = './meshlab/meshconv {} -c off -o {}'
os.system(off_tmp.format(args.input_path, args.out_path))
