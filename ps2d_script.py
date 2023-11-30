# -*- coding: utf-8 -*-
"""
Created on Thursday Nov 30 10:30:43 2023

@author: Ming Jiang (mingjiang@xidian.edu.cn)
"""


import numpy as np
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
import ps2d
import astropy.constants as ac
from os.path import join
import argparse

# HI line frequency
freq21cm = 1420.405751  # [MHz]


def freq2z(freq):
    z = freq21cm / freq - 1.0
    return z


def main(args):
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
    freq_start = args.f
    freq_end = args.g
    #
    # freq_start = 1810
    # freq_end = 1960

    dfreq = 0.1 # MHz
    frequencies = 0.1 * np.arange(freq_start, freq_end+1)  # MHz
    freqc = frequencies.mean()  # central frequency
    zc = freq2z(freqc)  # redshift
    print("Central frequency {:.2f} [MHz] <-> redshift {:.4f}".format(freqc, zc))

    destDir = "/media/xd/disk/Data/sdc3_data_challenge/results/"
    filename = join(destDir, f"eor_cube_f{freq_start/10:.2f}_{freq_end/10:.2f}.fits")
    cube = fits.getdata(filename)

    # Adopted cosmology
    H0 = 100 # [km/s/Mpc]
    OmegaM0 = 0.30964
    cosmo = FlatLambdaCDM(H0=H0, Om0=OmegaM0)

    pixelsize = 16
    pixelsize /= 3600  # [arcsec] -> [deg]

    DMz = cosmo.comoving_transverse_distance(zc).value
    d_xy = DMz * np.deg2rad(pixelsize)

    c = ac.c.to("km/s").value  # [km/s]
    Hz = cosmo.H(zc).value  # [km/s/Mpc]
    d_z = c * (1+zc)**2 * dfreq / Hz / freq21cm

    bin_edges_klos = np.arange(0.025, 0.55, 0.05) #[5.000000e-02, 1.000000e-01, 1.500000e-01, 2.000000e-01, 2.500000e-01, 3.000000e-01, 3.500000e-01, 4.000000e-01, 4.500000e-01, 5.000000e-01]
    bin_edges_kper = np.arange(0.025, 0.55, 0.05) #[5.000000e-02, 1.000000e-01, 1.500000e-01, 2.000000e-01, 2.500000e-01, 3.000000e-01, 3.500000e-01, 4.000000e-01, 4.500000e-01, 5.000000e-01]

    ps2d_y_rec,_,_=ps2d.calculate_2d_power_spectrum(cube,[d_z,d_xy], bin_edges_klos=bin_edges_klos, bin_edges_kper=bin_edges_kper)

    resfile = f"Hausos_{freq_start/10:.1f}MHz-{(freq_end+1)/10:.1f}MHz_errors.data"
    np.savetxt(resfile, ps2d_y_rec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', default='1060', type=int, help='start freq')
    parser.add_argument('-g', default='1209', type=int, help='end freq')
    args = parser.parse_args()
    print('Frequency: {:.1f} MHz to {:.1f} MHz'.format(args.f/10, args.g/10))
    main(args)