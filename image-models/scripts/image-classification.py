#!/usr/bin/env python
# coding: utf-8

# # Humanitarian Task Image Classification 
# 
# In this notebook, we use a pre-trained Inception v3 image classifier to classify tweet images for humanitarian task classification.

# In[256]:


import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
import csv
import math
import pickle
import stanza
import cv2
import copy
import seaborn as sns
import time
from PIL import Image


from torch.utils.data.sampler import BatchSampler, RandomSampler, Sampler,     SequentialSampler, SubsetRandomSampler
from torch.utils.data import Dataset
from torch.nn import functional
from torch import nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix,     precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, OrderedDict, Counter
from json import load, dump
from torchvision import datasets, models, transforms


# ## 1. Load Datafiles

# In[5]:


train_split_df = pd.read_csv('./data/crisis-mmd/task_humanitarian_text_img_train.tsv', sep='\t')


# In[6]:


train_split_df.head()


# In[11]:


labels = Counter(train_split_df['label'])
labels


# In[13]:


label_map = {
    'affected_individuals': 0,
    'infrastructure_and_utility_damage': 1,
    'not_humanitarian': 2,
    'other_relevant_information': 3,
    'rescue_volunteering_or_donation_effort': 4,
    'vehicle_damage': 1,
    'injured_or_dead_people': 0,
    'missing_or_found_people': 0
}


# In[82]:


def image_transform(image):

    h, w, _ = image.shape
    target_len = 299
    padded_image = np.zeros((target_len, target_len, 3)).astype(int)


    if w > h:
        # Resize to [] x 299
        image = cv2.resize(image, (target_len, math.floor(h / (w / target_len))))
        short_len = image.shape[0]

        # Pad to 299 x 299
        padded_image[math.floor((target_len - short_len) / 2):
                     math.floor((target_len - short_len) / 2) + short_len,
                     :, :] = image
    else:
        # Resize to 299 x []
        image = cv2.resize(image, (math.floor(w / (h / target_len)), target_len))
        short_len = image.shape[1]

        # Pad to 299 x 299
        padded_image[:, math.floor((target_len - short_len) / 2):
                     math.floor((target_len - short_len) / 2) + short_len,
                     :] = image
    
    return padded_image

image = cv2.imread('./data/crisis-mmd/{}'.format(train_split_df.iloc[3, 4]))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# In[84]:


plt.imshow(image)
plt.show()


# In[85]:


plt.imshow(image_transform(image))
plt.show()


# In[239]:


class ImageDataset(Dataset):

    def __init__(self, tsv_file, root_dir):
        """
        Args:
            tsv_file (string): Path to the train/test/dev split tsv file
            root_dir (string): Directory with all the images.
        """
        self.file_df = pd.read_csv(tsv_file, sep='\t')
        self.root_dir = root_dir
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.file_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.file_df.iloc[idx, 4])
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize one dimension of image to 299
        image = image_transform(image)
        
        # Swap axis
        image = image.transpose((2, 0, 1))
        
        # Normalize the image
        image = image / 255
        image = torch.from_numpy(image)
        image = self.normalize(image).float()
        
        # Get the label
        label = label_map[self.file_df.iloc[idx, 5]]
        
        return image, label


# In[240]:


train_dataset = ImageDataset('./data/crisis-mmd/task_humanitarian_text_img_train.tsv',
                             './data/crisis-mmd')


# In[241]:


for i in train_dataset:
    print(i[0].shape, i[1])
    break


# ## 2. Fine-tune Inception v3

# In[242]:


num_classes = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[243]:


train_dataset = ImageDataset('./data/crisis-mmd/task_humanitarian_text_img_train.tsv',
                             './data/crisis-mmd')
dev_dataset = ImageDataset('./data/crisis-mmd/task_humanitarian_text_img_dev.tsv',
                             './data/crisis-mmd')
test_dataset = ImageDataset('./data/crisis-mmd/task_humanitarian_text_img_test.tsv',
                             './data/crisis-mmd')


# In[244]:


batch_size = 32

train_dataset_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=4)
dev_dataset_loader = torch.utils.data.DataLoader(dev_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=4)
test_dataset_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=4)


# In[245]:


for inputs, labels in train_dataset_loader:
    print(inputs.shape, labels)
    break


# In[250]:


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25,
                patience=20, is_inception=True):
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Early stopping
    es_counter = 0
    es = False

    for epoch in range(num_epochs):
        if es:
            break

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    es_counter = 0
                else:
                    es_counter += 1
                    if es_counter > patience:
                        es = True

                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


# In[247]:


model_ft = models.inception_v3(pretrained=True)

for name, param in model_ft.named_parameters():
    
    if name not in set(['fc.weight', 'fc.bias', 'AuxLogits.fc.weight', 'AuxLogits.fc.bias']):
        param.requires_grad = False
    else:
        param.requires_grad = True


# In[296]:


# Handle the auxilary net
num_ftrs = model_ft.AuxLogits.fc.in_features
model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)

# Handle the primary net
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs,num_classes)

model_ft = model_ft.to(device)

params_to_update = [p for p in model_ft.parameters() if p.requires_grad]

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(params_to_update, lr=1e-6, weight_decay=0.001)
criterion = nn.CrossEntropyLoss()


# In[295]:


# dataloaders = {'train': train_dataset_loader,
#                'val': dev_dataset_loader}

# model, val_acc_history = train_model(model_ft, dataloaders, criterion,
#                                      optimizer_ft, num_epochs=500,
#                                      patience=50, is_inception=True)

# torch.save(model.state_dict(), './output/best_weights.pk')
# dump(val_acc_history, open('./output/val_acc_history.json', 'w'))


# ## VGG-16

# In[287]:


class ImageDataset(Dataset):

    def __init__(self, tsv_file, root_dir):
        """
        Args:
            tsv_file (string): Path to the train/test/dev split tsv file
            root_dir (string): Directory with all the images.
        """
        self.file_df = pd.read_csv(tsv_file, sep='\t')
        self.root_dir = root_dir
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.resize = transforms.Resize(256)
        self.center = transforms.CenterCrop(224)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.file_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.file_df.iloc[idx, 4])
        image = Image.open(img_name).convert('RGB')
        
        image = self.resize(image)
        image = self.center(image)
        
        image = self.to_tensor(image)
        image = self.normalize(image)
        
        # Get the label
        label = label_map[self.file_df.iloc[idx, 5]]
        
        return image, label


# In[288]:


num_classes = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataset = ImageDataset('./data/crisis-mmd/task_humanitarian_text_img_train.tsv',
                             './data/crisis-mmd')
dev_dataset = ImageDataset('./data/crisis-mmd/task_humanitarian_text_img_dev.tsv',
                             './data/crisis-mmd')
test_dataset = ImageDataset('./data/crisis-mmd/task_humanitarian_text_img_test.tsv',
                             './data/crisis-mmd')


# In[289]:


batch_size = 32

train_dataset_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=4)
dev_dataset_loader = torch.utils.data.DataLoader(dev_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=4)
test_dataset_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=4)


# In[290]:


for inputs, labels in train_dataset_loader:
    print(inputs.shape, torch.max(inputs), labels)
    break


# In[291]:


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, patience=20):
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Early stopping
    es_counter = 0
    es = False

    for epoch in range(num_epochs):
        if es:
            break

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    es_counter = 0
                else:
                    es_counter += 1
                    if es_counter > patience:
                        es = True

                val_acc_history.append(epoch_acc)

        print()

    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


# In[292]:


model = models.vgg16(pretrained=True)

# Freeze model weights
for param in model.parameters():
    param.requires_grad = False
    
# Add on classifier
model.classifier[6] = nn.Sequential(
                      nn.Linear(4096, 256), 
                      nn.ReLU(), 
                      nn.Dropout(0.4),
                      nn.Linear(256, num_classes))

model = model.to(device)

params_to_update = [p for p in model.parameters() if p.requires_grad]

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(params_to_update, lr=1e-4)
criterion = nn.CrossEntropyLoss()


# In[294]:


dataloaders = {'train': train_dataset_loader,
               'val': dev_dataset_loader}

model, val_acc_history = train_model(model, dataloaders, criterion,
                                     optimizer_ft, num_epochs=200,
                                     patience=20)

torch.save(model.state_dict(), './output/best_weights.pk')
dump(val_acc_history, open('./output/val_acc_history.json', 'w'))


# In[ ]:




