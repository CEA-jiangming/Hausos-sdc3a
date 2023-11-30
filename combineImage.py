# -*- coding: utf-8 -*-
"""
Created on Thursday Nov 30 10:30:43 2023

@author: Ming Jiang (mingjiang@xidian.edu.cn)
"""

import numpy as np
from astropy.io import fits
import os
from os.path import join
import sys
import argparse


# freq_start = 1060
# freq_end = 1209
# #
# freq_start = 1210
# freq_end = 1359
# #
# freq_start = 1360
# freq_end = 1509
# #
# freq_start = 1510
# freq_end = 1659
# #
# freq_start = 1660
# freq_end = 1809
#
freq_start = 1810
freq_end = 1960

Nx = 2048
Ny = 2048
ct = 2048
ps = 256
st = 128
resDir = "/media/xd/disk/Data/sdc3_data_challenge/results/freq_{}_{}".format(freq_start, freq_end)
num1 = (ct - ps) // st + 1
num2 = (ct - ps) // st + 1
freq = range(freq_start, freq_end+1)
cube = np.zeros((len(freq), Nx, Ny))
count = np.zeros((len(freq), Nx, Ny))

for ind, f in enumerate(freq):
    print(f"freqency: {f/10:.2f} MHz")
    for i in range(num1):
        for j in range(num2):
            filename = join(resDir, "msn_image_Temp_f{:.2f}_P{}_S{}_{}_{}_predict.fits".format(f/10, ps, st, i, j))
            patch = fits.getdata(filename).squeeze()
            cube[ind, i*st:i*st+ps, j*st:j*st+ps] += patch
            count[ind, i*st:i*st+ps, j*st:j*st+ps] += 1
            # cube[ind, i * st:i * st + ps, j * st:j * st + ps] = patch
            # if i*st % ps == 0 and j*st % ps == 0:
            #     cube[ind, i * st:i * st + ps, j * st:j * st + ps] = patch

# print("4-time-overlap region: {st}:{Nx-st}, {st}:{Ny-st}")
# cube[:, st:-st, st:-st] /= 4. # only for st = ps/2
# print("2-time-overlap region: 0/{Nx-st}:{st}/Nx,{st}:{Ny-st}")
# cube[:, :st, st:-st] /= 2.
# cube[:, -st:, st:-st] /= 2.
# cube[:, st:-st, :st] /= 2.
# cube[:, st:-st, -st:] /= 2.
cube = cube / count

destDir = "/media/xd/disk/Data/sdc3_data_challenge/results/"
fits.writeto(join(destDir, f"eor_cube_f{freq_start/10:.2f}_{freq_end/10:.2f}.fits"), cube)
