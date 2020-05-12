from __future__ import print_function, division
from torch.utils.tensorboard import SummaryWriter
import os
import time
import torch
import torch.utils.data
import torch.nn.functional as F
import torch.nn
import torchvision
from torch import optim
import torchsummary
import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt
import natsort
from torch.utils.data.sampler import SubsetRandomSampler
from Data_Loader import Images_Dataset, Images_Dataset_folder
import shutil
import random
from Models import Unet_dict, NestedUNet, U_Net, R2U_Net, AttU_Net, R2AttU_Net
from losses import *
from ploting import plot_kernels, LayerActivations, input_images, plot_grad_flow
from Metrics import dice_coeff, accuracy_score
from visdom import Visdom
from ploting import VisdomLinePlotter
from CreateFloder import *

#######################################################
#Checking if GPU is used
#######################################################
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available. Training on CPU')
else:
    print('CUDA is available. Training on GPU')
device = torch.device("cuda:0" if train_on_gpu else "cpu")

#######################################################
#Setting the basic paramters of the model
#######################################################
batch_size = 4
epoch = 9
random_seed = random.randint(1, 100)
print('epoch = ' + str(epoch))
print('batch_size = ' + str(batch_size))
print('random_seed = ' + str(random_seed))

shuffle = True
valid_loss_min = np.Inf
num_workers = 0
lossT = []
lossL = []
lossL.append(np.inf)
lossT.append(np.inf)
valid_size = 0.3
epoch_valid = epoch - 2
n_iter = 1
i_valid = 0

#If you load your samples in the Dataset on CPU and 
#would like to push it during training to the GPU, 
#you can speed up the host to device transfer by enabling pin_memory.
#This lets your DataLoader allocate the samples in page-locked memory, 
#which speeds-up the transfer.
pin_memory = False
if train_on_gpu:
    pin_memory = True

#######################################################
#Setting up the model
#######################################################
#model_Inputs = [U_Net, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet]
model_Inputs = [U_Net]

#整个模型的输入图片的channel
#以及最终经过模型处理后输出图片的channel
def model_unet(model_input, in_channel=3, out_channel=1):
    model_test = model_input(in_channel, out_channel)
    return model_test

numOfClass = 3
#创建模型
model_test = model_unet(model_Inputs[0], 1, numOfClass)
#迁移模型至GPU
model_test.to(device)

#######################################################
#Getting the Summary of Model
#######################################################
#torchsummary.summary(model_test, input_size=(3, 128, 128))

writer = SummaryWriter()

#createFloders(epoch, batch_size)

#######################################################
#Passing the Dataset of Images and Labels
#######################################################
#t_data = 'E:/Phd/unetdataset/train-s/'
#l_data = 'E:/Phd/unetdataset/train_masks-s/'

t_data = 'E:/Phd/UnetGF/data/membrane/train/image/'
l_data = 'E:/Phd/UnetGF/data/membrane/train/label/'

test_folderP = 'E:/Phd/unetdataset/'
test_folderL = 'E:/Phd/unetdataset/' 

#迭代器
#获得转换后的图片与标签
Training_Data = Images_Dataset_folder(t_data, l_data)
#######################################################
#Trainging Validation Split
#######################################################
num_train = len(Training_Data)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))

if shuffle:
    np.random.seed(random_seed)
    #Modify a sequence in-place by shuffling its contents
    np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]
#Samples elements randomly from a given list of indices, without replacement
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(Training_Data, batch_size=batch_size, \
                                           sampler=train_sampler,
                                           num_workers=num_workers, pin_memory=pin_memory,)

valid_loader = torch.utils.data.DataLoader(Training_Data, batch_size=1, \
                                           sampler=valid_sampler,
                                           num_workers=num_workers, pin_memory=pin_memory,)

#######################################################
#Using Adam as Optimizer
#######################################################
initial_lr = 0.001
opt = torch.optim.Adam(model_test.parameters(), lr= initial_lr) 
#opt = optim.SGD(model_test.parameters(), lr = initial_lr, momentum=0.99)

MAX_STEP = int(1e10)
#Set the learning rate of each parameter group using a cosine annealing schedule
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MAX_STEP, eta_min=1e-5)
#scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, epoch, 1)

#######################################################
#Training loop
#######################################################

offset_train = int(np.ceil(len(train_idx) / batch_size))

for i in range(epoch):
    train_loss = 0.0
    valid_loss = 0.0
    since = time.time()    

    #######################################################
    #Training Data
    #######################################################
    model_test.train()

    #print("模型训练开始")
    #time_elapsed = time.time()

    for itrain, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        #grid_img_x = torchvision.utils.make_grid(x)
        #writer.add_image('trainFeatureImagesGroup_{}'.format(offset_train * i + itrain), grid_img_x, 0) 
        #grid_img_y = torchvision.utils.make_grid(x)
        #writer.add_image('trainLabelImagesGroup_{}'.format(offset_train * i + itrain), grid_img_y, 0) 
        #time_elapsed = time.time() - time_elapsed
        #print('将数据迁移至CUDA: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        
        opt.zero_grad()
        y_pred = model_test(x)
        #time_elapsed = time.time() - time_elapsed
        #print('模型处理完数据: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        
        alpha, reaLabel = PretreatBefCalLoss(y_pred, y, numOfClass)
        lossT = torch.mean(DirichletLoss(reaLabel, alpha, numOfClass))
        
        
        #lossT = calc_loss(y_pred, y)     # Dice_loss Used
        writer.add_scalar('Loss-Train-Iter', lossT, offset_train * i + itrain)
        train_loss += lossT.item()

        #time_elapsed = time.time() - time_elapsed
        #print('计算完损失函数: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        
        #train_loss += lossT.item() * x.size(0)
        lossT.backward()
        opt.step()
        #time_elapsed = time.time() - time_elapsed
        #print('完成反向传播: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        
    #######################################################
    #Validation Step
    #######################################################
    model_test.eval()
    torch.no_grad() #to increase the validation process uses less memory

    accuracy = 0
    accuracy1 = 0
    for itest, (x1, y1) in enumerate(valid_loader):
        x1, y1 = x1.to(device), y1.to(device)
        pred_tb = model_test(x1).to(device).cpu().detach()

        alpha, reaLabel = PretreatBefCalLoss(pred_tb, y1, numOfClass)
        pred_tb = MaxmunVote(pred_tb)
        pred_tb = threshold_predictions_p(pred_tb)
        accuracy += accuracy_score(pred_tb[0][0].numpy(), y1.cpu().detach().numpy()[0])
        
        uncertainty = CalculateUncertainty(alpha, pred_tb.shape)
        predRefined = RefineWithUncertainty(pred_tb, y1, uncertainty)
        accuracy1 += accuracy_score(predRefined[0][0], y1.cpu().detach().numpy()[0])
        
    accuracy /= len(valid_idx)
    accuracy1 /= len(valid_idx)
    #Step could be called after every batch update
    scheduler.step(i)
    #lr = scheduler.get_lr()

    #######################################################
    #To write in Tensorboard
    #######################################################
    #计算平均损失大小
    train_loss = train_loss / len(train_idx)
    print('Epoch: {}/{} \tTraining Loss: {:.6f} \t  Acc: {:.6f}\t  Acc1: {:.6f}'.format(i + 1, epoch, \
        train_loss,  accuracy,  accuracy1))
    writer.add_scalar('Loss-Train-Epoch', train_loss, i)
    writer.add_scalar('Acc-Epoch', accuracy, i)
    writer.add_scalar('AccAfterRefine-Epoch', accuracy1, i)
    writer.add_image('Pred', pred_tb[0]) 
    
    #######################################################
    # 模型参数保存
    #######################################################
    #torch.save(model_test.state_dict(), './model/Unet_D_' +
    #                                          str(epoch) + '_' + str(batch_size) + '/Unet_epoch_' + str(epoch)
    #                                          + '_batchsize_' + str(batch_size) + '.pth')

    time_elapsed = time.time() - since
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    n_iter += 1

    torch.save(model_test.state_dict(), './model1/'  + str(numOfClass) + 'out_' + 
                                            str(len(train_idx)) +  'trainidx_' +
                                               str(batch_size) + 'batchsize_' +'.pth')

    #torch.save(model_test.state_dict(), 'model1/model.ptk')

#######################################################
#closing the tensorboard writer
#######################################################
writer.close()
