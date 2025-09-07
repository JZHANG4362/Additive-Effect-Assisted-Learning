###############################
#  cla = '2' response: speed  > 200
###############################

import numpy as np
import pandas as pd
import os
import glob
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision 
from torch.utils import data as D
import pickle
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from util import *
######################################################################################################################################################
# preparation.
######################################################################################################################################################

# ############################################################################################
# # Extract directory addresses of the img and label files
# ############################################################################################
class DConfig(object):
    parent_folder = './data/compcars/data/label/'
    attri = './data/compcars/data/misc/attributes.txt'

op = DConfig()
# obtain the file addressess in a folder
def listdir_nohidden(path):
    return glob.glob(os.path.join(path, '*'))

# # array to store attribute image addressess and their first attributes
# labelArray = np.empty((0, 3))

# parent_folder = op.parent_folder
# subfolders = listdir_nohidden(parent_folder)
# #i = 0
# for i in range(len(subfolders)):
#     print(i)
#     subsubfolders = listdir_nohidden(subfolders[i])
#     for j in range(len(subsubfolders)):
#     #j = 0
#         subsubsubfolders = listdir_nohidden(subsubfolders[j])

#         for l in range(len(subsubsubfolders)):
#         #l = 0
            
#             subsubsubsubfolders = listdir_nohidden(subsubsubfolders[l])
#             for k in range(len(subsubsubsubfolders)):
#             #k = 0
#                 with open(subsubsubsubfolders[k], 'r') as f:
#                     lines = f.read()
                
#                 # remove '\n' and space
#                 lines1 = lines.split('\n')

#                 # corresponding image address
#                 add1 = subsubsubsubfolders[k].replace('label', 'image')
#                 add2 = add1.replace('txt', 'jpg')

#                 # store the file address and attribute
#                 add_attri = [add2, subsubsubsubfolders[k], lines1[0]]


#                 labelArray = np.concatenate((labelArray, np.array(add_attri).reshape(1,-1)), axis = 0)


# # store the addresses and label
# pickle.dump(labelArray, open("imgDirectories.p", "wb"))

# ############################################################################################
# # Read the attribute file and drop the lines with missing 'speed'
# ############################################################################################
attriArray = np.empty((0, 6))
with open(op.attri, newline='\n') as trainfile:
    for line in trainfile:
        line3 = line.replace("\n","")
        line4 = line3.split(' ')
        attriArray = np.concatenate((attriArray, np.array(line4).reshape(1,-1)), axis = 0)

attriArray = attriArray[1:,:]

# # obtain the speed attributes for different models
# speed = attriArray[:,1][1:].astype(float)

# # np.median(speed[np.where(speed > 0)]) 
# sped = 200
# # delete those with speed unknown
# SpeedSelInd = attriArray[np.where(speed > 0), 0]

# lowSpeedInd = attriArray[np.where((speed > 0) & (speed < sped)),0]

# FastSpeedInd = attriArray[np.where(speed >= sped),0]


# # add model id
# labelArray2 = np.append(labelArray, np.zeros(labelArray.shape[0]).reshape(-1,1), axis = 1)
# for i in range(labelArray.shape[0]):
#     print(i)
#     labelArray2[i, 3] = labelArray[i,1].split('/')[6]

# # drop the rows with speed missing
# labelArray3 = np.empty((0, 4))
# for i in range(labelArray2.shape[0]):
#     print(i)
#     if labelArray2[i,3] in SpeedSelInd:
#         labelArray3 = np.concatenate((labelArray3, labelArray2[i,:].reshape(1,-1)), axis = 0)
# # store the selected addresses and label with non-missing speed
# pickle.dump(labelArray3, open("imgDirectories3.p", "wb"))

# # ######################################################################################################################################################
# # get the directory of paired datasets
# # ######################################################################################################################################################
# infile = open('imgDirectories3.p', 'rb')
# labelArray3 = pickle.load(infile)
# infile.close()


# apdArray = np.array(['none', 'none', 'none', 'none'])
# labelArray5 = np.concatenate((labelArray3, apdArray.reshape(1,-1)), axis = 0)


# # initialization for k=0

# k = 0
# resarray = np.empty((0,10))
# tmp = labelArray5[k,:][0].split('/')
# addnew = "_".join(tmp[5:8])
# angle = labelArray5[k,2]

# tmp_array = np.array(['none', 
#                         'none', 'none', 
#                         'none', 'none',
#                         'none', 'none',
#                         'none', 'none',
#                         'none'], dtype=object)
# tmp_array[0] = addnew
# tmp_array[1] = labelArray5[0,2]
# tmp_array[2] = labelArray5[0,3]
# tmp_array[9] = int(attriArray[np.where(attriArray[:,0] == labelArray3[0,3])[0],1])
# addold = addnew

# if (labelArray5[k,2] == '1') and tmp_array[3] == 'none':
#     tmp_array[3] = labelArray5[k,0]
#     tmp_array[4] = labelArray5[k,1]
# if (labelArray5[k,2] == '3') and tmp_array[5] == 'none':
#     tmp_array[5] = labelArray5[k,0]
#     tmp_array[6] = labelArray5[k,1]
# if (labelArray5[k,2] == '2') and tmp_array[7] == 'none':
#     tmp_array[7] = labelArray5[k,0]
#     tmp_array[8] = labelArray5[k,1]


# # counter of tmp_array array
# ct = 0
# for k in range(1,labelArray5.shape[0]):
# #for k in range(1,600):
#     print(k)

#     tmp = labelArray5[k,:][0].split('/')
#     addnew = "_".join(tmp[5:8])
#     # if the iteration proceeds to a new subfolder
#     if addnew != addold:
#         ct += 1
#         # if the previous subfolder has both images with angles '1' and '3', add tmp_array to resarray
#         if 'none' not in tmp_array:
#             resarray = np.concatenate((resarray, tmp_array.reshape(1,-1)), axis = 0)
#         # set tmp_array to default
#         tmp_array = np.array(['none', 
#                         'none', 'none', 
#                         'none', 'none',
#                         'none', 'none',
#                         'none', 'none',
#                         'none'], dtype=object)
#         # add folder address to tmp_array
#         tmp_array[0] = addnew
#         # add the other attributes
#         tmp_array[1] = labelArray5[k,2]
#         tmp_array[2] = labelArray5[k,3]
#         if k < (labelArray5.shape[0] - 1):
#             tmp_array[9] = int(attriArray[np.where(attriArray[:,0] == labelArray3[k,3])[0],1])

#     if (labelArray5[k,2] == '1') and tmp_array[3] == 'none':
#         tmp_array[3] = labelArray5[k,0]
#         tmp_array[4] = labelArray5[k,1]
#     if (labelArray5[k,2] == '3') and tmp_array[5] == 'none':
#         tmp_array[5] = labelArray5[k,0]
#         tmp_array[6] = labelArray5[k,1]
#     if (labelArray5[k,2] == '2') and tmp_array[7] == 'none':
#         tmp_array[7] = labelArray5[k,0]
#         tmp_array[8] = labelArray5[k,1]
#     # print(tmp_array)
#     # print(labelArray5[k,2])
#     # store the odd address variable
#     addold = addnew

# labelArray_label123 = resarray
# labelArray_label1 = resarray[:,[0,1, 2,3,4,9]]

# labelArray_label3 = resarray[:,[0,1, 2,5, 6,9]]

# labelArray_label2 = resarray[:,[0,1, 2,7, 8,9]]

# subsetRes = {'labelArray_label123' : labelArray_label123, 'labelArray_label1' : labelArray_label1, 'labelArray_label3' : labelArray_label3, 'labelArray_label2' : labelArray_label2}

# #####################################################################################################################################################
# #store the result
# pickle.dump(subsetRes, open("subsetRes4.p", "wb"))

# ######################
infile = open('subsetRes4.p', 'rb')
subsetRes = pickle.load(infile)
infile.close()

labelArray_label1 = subsetRes['labelArray_label1']
labelArray_label3 = subsetRes['labelArray_label3']
labelArray_label2 = subsetRes['labelArray_label2']
######################################################################################################################################################
cla = '2'
if cla == '1':
    labelArray4 = labelArray_label1
elif cla == '3':
    labelArray4 = labelArray_label3
elif cla == '2':
    labelArray4 = labelArray_label2
else:
    print('error')
# the response label is whether the max speed of the car is larger than 'sped'
sped = 200

group1Ind = np.where(labelArray4[:,5] > sped)[0]
group2Ind = np.where(labelArray4[:,5] <= sped)[0]

# split the image indices into training and validation
np.random.seed(10)
shuffleInd1 = np.arange(group1Ind.shape[0])
np.random.shuffle(shuffleInd1)

shuffleInd2 = np.arange(group2Ind.shape[0])
np.random.shuffle(shuffleInd2)

halfNum1 = round(shuffleInd1.shape[0]/2)
group1IndTrain, group1IndVal = group1Ind[shuffleInd1[:halfNum1]], group1Ind[shuffleInd1[halfNum1]:]

halfNum2 = round(shuffleInd2.shape[0]/2)
group2IndTrain, group2IndVal = group2Ind[shuffleInd2[:halfNum2]], group2Ind[shuffleInd2[halfNum2]:]

# combine the splitted indices for response = 1 and response = 0
TrainInd = np.concatenate((group1IndTrain, group2IndTrain))
ValInd = np.concatenate((group1IndVal, group2IndVal))

###############################
#  Define the dataset object. Apply necessary transform for alexnet
###############################
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
    	# new trial
    	# transforms.RandomPerspective(),
        # classical
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
class CompcarsDS(D.Dataset):
    """
    A customized data loader.
    """
    def __init__(self, Ind, mode):
        """ Intialize the dataset
        """

        self.filenames = labelArray4[Ind,3]

        self.len = len(self.filenames)
        self.transform = data_transforms[mode]
                           
    # You must override __getitem__ and __len__
    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        image = Image.open(self.filenames[index])
        model_id = self.filenames[index].split("/")[6]
        # obtain the car speed
        label = int(int(attriArray[np.where(attriArray[:,0] == model_id),1][0,0]) > sped)
        return self.transform(image), label
    
    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len


# prepare train val datasets and data loaders
datTrain = CompcarsDS(Ind = TrainInd, mode = 'train')
datVal= CompcarsDS(Ind = ValInd, mode = 'val')
image_datasets = {'train': datTrain, 'val': datVal}

batch_size = 6

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
print('dataset_sizes: ', dataset_sizes)


d = 15 # the final output dim, so the architecture for mobilenetv2 is 1280 - d (retrain) - 2 (retrain)


###########################################
# define and train transfer model
##########################################
model = torchvision.models.mobilenet_v2(pretrained=True) #used for cat-dog
for param in model.parameters():
    param.requires_grad = False
model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.last_channel, d),
        )
model.fc2 = nn.Linear(d, 2)
# update only these parameters
D_parameters = [
    {'params': model.classifier.parameters()},
    {'params': model.fc2.parameters()}
]



# Observe that only parameters of final layer are being optimized as
# optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(D_parameters, lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = train_model(dataloaders, dataset_sizes,
	model, criterion, optimizer, exp_lr_scheduler, num_epochs=1)



def extract_feature(dataloader, model, d, savefile):
	model.eval()      
	X = np.zeros((0, d))
	y = np.zeros(0)
	for i, (inputs, labels) in enumerate(dataloader):
		print('batch ', i)

		f = model(inputs) 
		# print('feature size: ', f.size())
		batchsize = labels.size()[0]
		f = f.view(batchsize, -1).data.numpy()
		labels = labels.data.numpy()
		# print('reshaped feature size (batch_size x feature dim): ', f.shape)
		# print(f)
		X = np.concatenate((X, f), axis=0)
		y = np.concatenate((y, labels), axis=0)

	yX = np.concatenate((y.reshape((-1,1)), X), axis=1)
	print('responses and features \n', yX)
	print('size: ', yX.shape)
	np.savetxt(savefile, yX, delimiter=",")


outputTrain = CompcarsDS(Ind = TrainInd, mode = 'val')
outputVal = CompcarsDS(Ind = ValInd, mode = 'val')

dataloaderTrain = torch.utils.data.DataLoader(outputTrain, batch_size=batch_size, shuffle=False)
dataloaderVal = torch.utils.data.DataLoader(outputVal, batch_size=batch_size, shuffle=False)

extract_feature(dataloaderTrain, model, d, "TrainSpeed_" + cla + ".csv")
extract_feature(dataloaderVal, model, d, "ValSpeed_" + cla + ".csv")