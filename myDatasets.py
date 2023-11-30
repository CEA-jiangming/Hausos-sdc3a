import random

import torch
import torch.nn as nn
from os.path import exists, join
from torchvision.transforms import Compose, CenterCrop, ToTensor, Normalize#, Scale
import torch.utils.data as data
from os import listdir
import numpy as np

import astropy.io.fits as fits

train_image_dir='/media/xd/disk/Data/sdc3'
train_tempfits_dir = 'data/image_briggs/dirty_patch'
train_eor_dir = 'skymap/eor_patch'

test_image_dir='/media/xd/disk/Data/sdc3_data_challenge'
test_tempfits_dir = 'ZW3_msn_patch'

def myDataset(dest):
    if not exists(dest):
        print("dataset not exist ")
    return dest


def input_transform():  # need to add data augmentation
    # return Compose([CenterCrop((1024, 1024)), ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return Compose([ToTensor()])


def target_transform():
    return Compose([ToTensor()])


def get_test_set():
    pass


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir=train_image_dir, action='train', freq_start=1060, freq_end=1960, input_transform=input_transform(), target_transform=target_transform()):
        super(DatasetFromFolder, self).__init__()
        self.tempfits_dir = train_tempfits_dir
        self.eor_dir = train_eor_dir
        self.tempfits_filenames_training = []
        self.tempfits_filenames_test = []
        self.eor_filenames_training = []
        self.eor_filenames_test = []
        for freq in range(freq_start, freq_end+1):
            tmp_file = [x for x in sorted(listdir(join(image_dir, self.tempfits_dir))) if str(float(freq/10)) in x]
            self.tempfits_filenames_training += sorted(random.sample(tmp_file, 40))
            for tmp_file_name in tmp_file:
                if tmp_file_name in self.tempfits_filenames_training:
                    split_ = tmp_file_name.strip().split("_")
                    frequency = split_[1].split('-')[0]
                    cut_no = split_[-1]
                    self.eor_filenames_training.append(f'deltaTb_f{frequency}_N2048_fov9.1_P256_S128_{cut_no}')
                else:
                    self.tempfits_filenames_test.append(tmp_file_name)
                    split_ = tmp_file_name.strip().split("_")
                    frequency = split_[1].split('-')[0]
                    cut_no = split_[-1]
                    self.eor_filenames_test.append(f'deltaTb_f{frequency}_N2048_fov9.1_P256_S128_{cut_no}')
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.image_dir = image_dir
        self.action = action

    def __getitem__(self, index):

        if self.action == 'train':
            tempfits_training = fits.getdata(join(self.image_dir, self.tempfits_dir, self.tempfits_filenames_training[index]))
            eor_training = fits.getdata(join(self.image_dir, self.eor_dir, self.eor_filenames_training[index]))
            tempfits_training = tempfits_training.squeeze().astype(np.float32)
            eor_training = eor_training.astype(np.float32)
            if self.input_transform:
                tempfits_training = self.input_transform(tempfits_training)
            if self.target_transform:
                eor_training = self.target_transform(eor_training)
            filename_tempfits = self.tempfits_filenames_training[index]
            filename_eor = self.eor_filenames_training[index]
            return tempfits_training, eor_training, filename_tempfits, filename_eor
        else:
            tempfits_test = fits.getdata(join(self.image_dir, self.tempfits_dir, self.tempfits_filenames_test[index]))
            eor_test = fits.getdata(join(self.image_dir, self.eor_dir, self.eor_filenames_test[index]))
            tempfits_test = tempfits_test.squeeze().astype(np.float32)
            eor_test = eor_test.astype(np.float32)
            if self.input_transform:
                tempfits_test = self.input_transform(tempfits_test)
            if self.target_transform:
                eor_test = self.target_transform(eor_test)
            filename_tempfits = self.tempfits_filenames_test[index]
            filename_eor = self.eor_filenames_test[index]
            return tempfits_test, eor_test, filename_tempfits, filename_eor


    def __len__(self):
        if self.action == 'train':
            return len(self.tempfits_filenames_training)
        else:
            return len(self.tempfits_filenames_test)



class testDatasetFromFolder(data.Dataset):
    def __init__(self, image_dir=test_image_dir, freq_start=1060, freq_end=1960, input_transform=input_transform(), target_transform=target_transform()):
        super(testDatasetFromFolder, self).__init__()
        self.tempfits_dir = test_tempfits_dir
        self.tempfits_filenames = []
        for freq in range(freq_start, freq_end+1):
            tmp_file = [x for x in sorted(listdir(join(image_dir, self.tempfits_dir))) if str(float(freq/10)) in x]
            self.tempfits_filenames += sorted(tmp_file)
        self.input_transform = input_transform
        self.image_dir = image_dir

    def __getitem__(self, index):

        tempfits = fits.getdata(join(self.image_dir, self.tempfits_dir, self.tempfits_filenames[index]))
        tempfits = tempfits.squeeze().astype(np.float32)
        if self.input_transform:
            tempfits = self.input_transform(tempfits)
        filename_tempfits = self.tempfits_filenames[index]
        return tempfits, filename_tempfits


    def __len__(self):
        return len(self.tempfits_filenames)