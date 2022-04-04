#!/usr/bin/env python
# coding: utf-8

# ```
# Multimodal sentiment detection dataset.
# 
# the files are here: https://www.dropbox.com/s/hpkdyuui5adqsek/multimodal_sentiment_dataset.zip?dl=0
# the dataset comes from this paper: https://arxiv.org/abs/1708.02099
# There are four classes ['creepy', 'gore', 'happy', 'rage'} (it'd be good if you could maintain this order as well)
# The dropbox also contains the train, val, and test split. The images are in the subdirectory called images , and are indexed as per the id in the .pkl files. For instance if an example has id <n>, the corresponding image would be ./images/<n>.jpg/png/jpeg (the file format may vary for some of the images)
# We are reporting the macro F1, precision, recall (and accuracy) – just like we did for the crisis dataset
# 
# Multimodal fakenews detection dataset.
#  
# the files are here: https://www.dropbox.com/s/esnr2ltp0z2ibq8/multmodal_fakenews_dataset.zip?dl=0
# the dataset comes from this paper: https://ojs.aaai.org/index.php/AAAI/article/view/7230
# There are two classes: 0 (not fake) and 1 (fake). Since this is binary classification task, we don't need to report macro-averaged scores. We can report the F1, precision and recall for the target class (i.e., 1/fake). In sklearn it can be specified by specifying average = 'binary' .
# The dropbox contains the images as well as the .pkl files for train and test set. There's no validation set, but I used 10% randomly sampled examples from the train set as my validation set. Also, please use the image_filename str (5vthXfxXiBvcdhZRirjpnhDJJi14UqtR in the following example)  as the id for any given example.
# {'title': "Teen Mom Star Jenelle Evans' Wedding Dress Is Available Here for $2999", 'image': '5vthXfxXiBvcdhZRirjpnhDJJi14UqtR.jpg', 'label': 0}
# ```

# In[2]:


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
from sklearn import metrics

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


# In[3]:


train_data = pickle.load(open('./data/final_dataset_train.pkl', 'rb'))
vali_data = pickle.load(open('./data/final_dataset_val.pkl', 'rb'))
test_data = pickle.load(open('./data/final_dataset_test.pkl', 'rb'))

encoder = {'creepy': 0, 'gore': 1, 'happy': 2, 'rage': 3}

train_data_pairs = [[d['image'], encoder[d['label']]] for d in train_data]
val_data_pairs = [[d['image'], encoder[d['label']]] for d in train_data]
test_data_pairs = [[d['image'], encoder[d['label']]] for d in train_data]


# In[4]:


class ImageDataset(Dataset):

    def __init__(self, data_pairs, root_dir):
        """
        Args:
            tsv_file (string): Path to the train/test/dev split tsv file
            root_dir (string): Directory with all the images.
        """
        self.data_pairs = data_pairs
        self.root_dir = root_dir
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.resize = transforms.Resize(256)
        self.center = transforms.CenterCrop(224)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.data_pairs[idx][0]) + '.jpg'
        
        try:
            image = Image.open(img_name).convert('RGB')
        except:
            idx = 0
            img_name = os.path.join(self.root_dir, self.data_pairs[0][0]) + '.jpg'
            image = Image.open(img_name).convert('RGB') 
        
        image = self.resize(image)
        image = self.center(image)
        
        image = self.to_tensor(image)
        image = self.normalize(image)
        
        # Get the label
        label = self.data_pairs[idx][1]
        
        return image, label


# In[5]:


train_dataset = ImageDataset(train_data_pairs, './data/images')
val_dataset = ImageDataset(val_data_pairs, './data/images')
test_dataset = ImageDataset(test_data_pairs, './data/images')


# In[6]:


batch_size = 32

train_dataset_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=4)
val_dataset_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=4)
test_dataset_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=4)


# In[7]:


for inputs, labels in train_dataset_loader:
    print(inputs.shape, torch.max(inputs), labels)
    break


# In[8]:


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


# In[9]:


num_classes = 4
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

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


# In[10]:


dataloaders = {'train': train_dataset_loader,
               'val': val_dataset_loader}

model, val_acc_history = train_model(model, dataloaders, criterion,
                                     optimizer_ft, num_epochs=100,
                                     patience=20)

torch.save(model.state_dict(), './output/best_weights_sentiment.pk')
# dump(val_acc_history, open('./output/val_acc_history.json', 'w'))


# In[ ]:




