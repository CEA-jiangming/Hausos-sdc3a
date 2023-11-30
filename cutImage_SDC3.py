# -*- coding: utf-8 -*-
"""
Created on Thursday Nov 30 10:30:43 2023

@author: Ming Jiang (mingjiang@xidian.edu.cn)
"""

import numpy as np
from astropy.io import fits
import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Cut an image into patches")
    parser.add_argument("-C", "--center", dest="ct", type=int, help="center size")
    parser.add_argument("-P", "--patch", dest="ps", type=int, help="patch size")
    parser.add_argument("-S", "--stride", dest="st", type=int, help="incremental step of patches")
    parser.add_argument("infile",
                        help="input FITS image file")
    parser.add_argument("outfile",
                        help="output prefix filename of the converted image")
    args = parser.parse_args()

    data = fits.getdata(args.infile).squeeze()
    Nz, Nx, Ny = np.shape(data)
    data1 = data[:, Nx//2-args.ct//2:Nx//2+args.ct//2, Ny//2-args.ct//2:Ny//2+args.ct//2]
    num1 = (args.ct - args.ps) // args.st + 1
    num2 = (args.ct - args.ps) // args.st + 1
    
    freq1 = 106
    freq2 = 196.1
    step = 0.1
    freq = np.arange(freq1, freq2, step)
    bands = len(freq)
    for i in range(num1):
        for j in range(num2):
            patch = data1[:, i*args.st:i*args.st+args.ps, j*args.st:j*args.st+args.ps]
            print("Creat patch ({}, {})".format(i, j))
            for b in range(bands):
                # print("Creat patches ({}, {}) of frequency {:.2f}".format(i, j, freq[b]))
                fits.writeto("{}_f{:.2f}_P{}_S{}_{}_{}.fits".format(args.outfile, freq[b], args.ps, args.st, i, j), patch[b])


if __name__ == "__main__":
    main()
