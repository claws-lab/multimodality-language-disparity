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

labels = load(open('./MMHS150K_GT.json', 'r'))

binary_labels = {}

for k in labels:
    binary_labels[k] = int(labels[k]['labels'].count(0) <= 1)
    
splits = {}

for name in ['train', 'val', 'test']:
    with open('./splits/{}_ids.txt'.format(name)) as fp:
        lines = [l.replace('\n', '') for l in fp.readlines()]
        splits[name] = lines
        
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


class ImageDataset(Dataset):

    def __init__(self, ids, root_dir):
        """
        Args:
            tsv_file (string): Path to the train/test/dev split tsv file
            root_dir (string): Directory with all the images.
        """
        self.ids = ids
        self.root_dir = root_dir
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.resize = transforms.Resize(256)
        self.center = transforms.CenterCrop(224)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, '{}.jpg'.format(self.ids[idx]))
        image = Image.open(img_name).convert('RGB')
        
        image = self.resize(image)
        image = self.center(image)
        
        image = self.to_tensor(image)
        image = self.normalize(image)
        
        # Get the label
        label = binary_labels[self.ids[idx]]
        
        return image, label
    
train_dataset = ImageDataset(splits['train'], './mmhs150k-img')

num_classes = 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataset = ImageDataset(splits['train'], './mmhs150k-img')
dev_dataset = ImageDataset(splits['val'], './mmhs150k-img')
test_dataset = ImageDataset(splits['test'], './mmhs150k-img')

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

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, patience=20):
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_f1 = 0
    
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
            
            running_predicts = []
            running_labels = []

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
                
                running_predicts.extend(preds.tolist())
                running_labels.extend(labels.data.tolist())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val':
                # Compute F1 score
                y_true = torch.Tensor(running_labels)
                y_pred = torch.Tensor(running_predicts)
                tp = (y_true * y_pred).sum().to(torch.float32)
                tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
                fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
                fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

                epsilon = 1e-7

                precision = tp / (tp + fp + epsilon)
                recall = tp / (tp + fn + epsilon)

                f1 = 2 * (precision*recall) / (precision + recall + epsilon)
                
                print(f1)
    
                if f1 > best_f1:
                    best_f1 = f1
                    best_model_wts = copy.deepcopy(model.state_dict())
                    es_counter = 0
                else:
                    es_counter += 1
                    if es_counter > patience:
                        es = True

                val_acc_history.append(epoch_acc)

        print()

    print('Best val f1: {:4f}'.format(best_f1))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

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

dataloaders = {'train': train_dataset_loader,
               'val': dev_dataset_loader}

model, val_acc_history = train_model(model, dataloaders, criterion,
                                     optimizer_ft, num_epochs=200,
                                     patience=10)

torch.save(model.state_dict(), './output/best_weights_mmhs150k_627.pk')
# dump(val_acc_history, open('./output/val_acc_history.json', 'w'))
