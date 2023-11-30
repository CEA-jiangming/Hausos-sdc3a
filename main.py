# -*- coding: utf-8 -*-
"""
Created on Thursday Nov 30 10:30:43 2023

@author: Ming Jiang (mingjiang@xidian.edu.cn)
"""

from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from myModels.setr import SETR
from myDatasets import DatasetFromFolder, testDatasetFromFolder, get_test_set
import torchvision
import os
from os.path import join
import time
from astropy.io import fits
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dest = os.path.abspath(os.path.dirname(__file__))

# algorithm-related parameters
image_size = 256        # image is cut into patches
batch_size = 1
epochs = 100
learning_rate = 1e-5    # learning rate for the deep learning method

# the six bands to be processed,
# the start and end frequency is multiplied by 10 for easy operation afterwards
freq_start = 1060
freq_end = 1209
#
# freq_start = 1210
# freq_end = 1359
#
# freq_start = 1360
# freq_end = 1509
#
# freq_start = 1510
# freq_end = 1659
#
# freq_start = 1660
# freq_end = 1809
#
# freq_start = 1810
# freq_end = 1960

model_name = 'model_setr'
model = SETR(image_size=image_size, patch_size=image_size//16, channels=1).to(device)
#

# one can select 'train' for training stage, 'testZW3' for testing stage
# or 'all' for both training and testing stages
action = 'all'
# action = 'train'
# action = 'testZW3'

obj = f'{model_name}_epochs{epochs}_freq{freq_start}_{freq_end}'

if action == 'testZW3':
    obj = f'ZW3_{model_name}_epochs{epochs}_freq{freq_start}_{freq_end}'

# criterion = torch.nn.MSELoss()
criterion = torch.nn.MSELoss()
criterion_dist = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=utils.LambdaLR(epochs, 0, int(epochs / 2)).step)


def checkpoint(epoch):
    os.makedirs('checkpoint_{}'.format(obj), exist_ok=True)
    model_out_path = join(f'checkpoint_{obj}', f'epoch{epoch+1}.pth')
    torch.save(model.state_dict(), model_out_path)
    # print("Checkpoint saved to {}".format(model_out_path))


def dice_loss(input, target):
    num = input * target
    num = torch.sum(num, dim=2)
    num = torch.sum(num, dim=2)
    den1 = input * input
    den1 = torch.sum(den1, dim=2)
    den1 = torch.sum(den1, dim=2)
    den2 = target * target
    den2 = torch.sum(den2, dim=2)
    den2 = torch.sum(den2, dim=2)
    dice = 2 * (num / (den1 + den2))
    dice_total = 1 - 1 * torch.sum(dice) / dice.size(0)  # divide
    return dice_total


def train():
    with open(join(dest, f'{model_name}_epoch{epochs}_freq{freq_start}_{freq_end}.csv'), 'w') as file:
        file.write('epoch,loss,dist\n')

    print('===> Loading training datasets')
    # logger.info('===> Loading training datasets')
    training_data = DatasetFromFolder(action='train', freq_start=freq_start, freq_end=freq_end)
    training_dataset_loader = DataLoader(dataset=training_data, num_workers=0, batch_size=batch_size, shuffle=False)

    loss_ave = []
    acc_ave = []
    print('===> Training '+model_name)
    # logger.info('===> Training '+model_name)
    total_time = 0
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))
        dataset_size = len(training_dataset_loader.dataset)
        epoch_loss = 0
        step = 0
        time_elapse = 0
        epoch_acc = 0
        epoch_dist = 0
        for tempfits, eor, filename_tempfits, filename_eor in training_dataset_loader:
            # print(f'{filename_tempfits} - {filename_eor}')
            start = time.time()
            tempfits = tempfits.to(device)
            eor = eor.to(device)

            optimizer.zero_grad()  # 清空梯度

            outputs = model(tempfits)

            loss = 0
            if 'plus2' in model_name:
                for i in range(len(outputs)):
                    loss += criterion(outputs[i], eor) * 100
            elif 'setr' in model_name:
                loss = criterion(outputs[-1], eor) + 0.2*criterion(outputs[0], eor) + 0.3*criterion(outputs[1], eor) + 0.4*criterion(outputs[2], eor)
            else:
                loss = criterion(outputs, eor) * 100

            loss.backward() # 计算梯度
            optimizer.step() # 反向传播， 更新网络参数
#           scheduler.step()
            epoch_loss += loss.item()
            step += 1
            max_arrow = 30
            prevL = step*max_arrow//((dataset_size // batch_size))
            nextL = max_arrow-prevL
            end = time.time()
            time_elapse += end-start
            # pred = outputs.argmax(dim=1)
            # pred = outputs
            # acc = torch.eq(pred.float(), eor.reshape(batch_size, image_size, image_size)).sum().float().item()/image_size/image_size/batch_size
            # epoch_acc += acc
            # dist = (pred.float() - eor.reshape(batch_size, image_size, image_size)).sum()
            dist = 0
            if 'plus2' in model_name:
                for i in range(len(outputs)):
                    dist += criterion_dist(outputs[i], eor).item() * 100
                dist /= len(outputs)
            elif 'setr' in model_name:
                dist += criterion_dist(outputs[-1], eor).item() * 100
            else:
                dist = criterion_dist(outputs, eor).item() * 100
            epoch_dist += dist
            print("\r" + f"{step}/{dataset_size // batch_size} [" + "="*prevL + "-"*nextL + f"] - loss: {loss.item():.6f}" + f" - dist: {dist:.6f}" + f" - {time_elapse:.0f}s", end=' ')
        print(f"\n{time_elapse/(dataset_size // batch_size):.0f}s/step - loss_ave: {epoch_loss /(dataset_size // batch_size):.6f} - dist_ave: {epoch_dist/(dataset_size // batch_size):.6f}")
        loss_ave.append(epoch_loss/(dataset_size // batch_size))
        acc_ave.append(epoch_acc/(dataset_size // batch_size))
        with open(join(dest, f'{model_name}_epoch{epochs}_freq{freq_start}_{freq_end}.csv'), 'a') as file:
            file.write(f'{epoch + 1},{epoch_loss / (dataset_size // batch_size):.6f},{epoch_dist / (dataset_size // batch_size):.6f}\n')
        total_time += time_elapse
        if (epoch+1) % epochs == 0:
            checkpoint(epoch)
        lr_scheduler.step()
    print("\nIt takes {:.2f} h for the training.\n".format(total_time/3600))
    # logger.info("\nIt takes {:.2f} h for the training.\n".format(total_time/3600))
    with open(join(dest, f'{model_name}_epoch{epochs}_acc-loss_freq{freq_start}_{freq_end}.csv'), 'a') as file:
        file.write('loss: \n{}\nacc: \n{}'.format(str(loss_ave),str(acc_ave)))
    torch.save(model.state_dict(), join(dest, f'{model_name}_epoch{epochs}_freq{freq_start}_{freq_end}.pth'))
    import matplotlib.pyplot as plt
    plt.plot(loss_ave, label='average loss for every epoch')
    plt.legend()
    plt.savefig(join(dest, f'{model_name}_epoch{epochs}_freq{freq_start}_{freq_end}.png'),dpi=600)


def testZW3():
    print('===> Loading test datasets')
    # logger.info('===> Loading test datasets')
    test_data = testDatasetFromFolder(freq_start=freq_start, freq_end=freq_end)
    test_dataset_loader = DataLoader(dataset=test_data, num_workers=0, batch_size=batch_size, shuffle=False)
    dataset_size = len(test_dataset_loader.dataset)
    if action == 'testZW3':
        weight = join(dest, f'{model_name}_epoch{epochs}_freq{freq_start}_{freq_end}.pth')
        model.load_state_dict(torch.load(weight,map_location='cpu'))
    # model.eval()
#    import matplotlib.pyplot as plt
#    plt.ion()
    with torch.no_grad():
        step = 0
        time_elapse = 0
        epoch_dist = 0
        for tempfits, filename_tempfits in test_dataset_loader:
            # print(f'{filename_tempfits}')
            tempfits = tempfits.to(device)
            start = time.time()
            step += 1
            predictions = model(tempfits)
            # dist = (predictions.float() - eor.reshape(batch_size, image_size, image_size)).sum()
            predictions_ave = predictions
            dist = 0
            if 'plus2' in model_name:
                predictions_ave = predictions[0]
                for i in range(1, len(predictions)):
                    predictions_ave += predictions[i]
                dist /= len(predictions)
                predictions_ave /= len(predictions)
            elif 'setr' in model_name:
                predictions_ave = predictions[-1]
            objRes = f"/media/xd/disk/Data/sdc3_data_challenge/results/freq_{freq_start}_{freq_end}"
            fits.writeto(join(objRes, filename_tempfits[0].split('.fits', 1)[0]+"_predict.fits"), predictions_ave.cpu().numpy())
            end = time.time()
            time_elapse += end-start
            max_arrow = 30
            prevL = step*max_arrow//dataset_size
            nextL = max_arrow-prevL
            print("\r" + f"{step}/{dataset_size} [" + "="*prevL + "-"*nextL + f"] - dist: {dist:.6f} - {time_elapse:.0f}s",end=' ')
        print(" - {:.0f}s/step".format(time_elapse/dataset_size))


if __name__ == '__main__':
    print('parameters:\nmodel: {} \nbatch_size: {} \nepochs: {}\nlearning_rate: {}'.format(model_name,batch_size,epochs,learning_rate))

    if action == 'train':
        train()
    elif action =='testZW3':
        os.makedirs(obj, exist_ok=True)
        testZW3()
    else:
        train()
        os.makedirs(obj, exist_ok=True)
        testZW3()
    # sys.stdout.close()

