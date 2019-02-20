# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 11:53:26 2018

@author: WT
"""
from GAN_net import D_net, G_net
import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn as nn
from torch import optim
import torch
import shutil

basepath = "./data/_12_Feb19/"

### open train files and compile into df, then save

def resize_images(size=224):
    for idx, file in enumerate(os.listdir(basepath)):
        try:
            imagename = os.path.join(basepath,file)
            img = Image.open(imagename)
            img = img.resize(size=(size,size))
            img.save(imagename)
        except:
            print(f"Image open error {idx}")
            continue
        
def compile_images():
    dataset = []
    labels = []
    for idx,file in enumerate(os.listdir(basepath)):
        try:
            imagename = os.path.join(basepath,file)
            img = Image.open(imagename)
            img = np.array(img)
            if img.shape == (224, 224, 3):
                dataset.append(img)
                labels.append(1)
        except:
            print(f"Image compile error {idx}")
            continue
    return dataset, labels

class birds_dataset(Dataset):
    def __init__(self, dataset, labels, transform=None):
        self.X = dataset
        self.y = labels
        self.transform = transform
    def __len__(self):
        return(len(self.y))
    def __getitem__(self,idx):
        img = self.X[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.y[idx]

def model_eval(net, test_loader, cuda=None):
    correct = 0
    total = 0
    with torch.no_grad():
        net.eval()
        for data in test_loader:
            images, labels = data
            if cuda:
                images, labels = images.cuda(), labels.cuda()
            images = images.float()
            labels = labels.long()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("Accuracy of the network on the 10000 test images: %d %%" % (
            100*correct/total))
    return 100*correct/total

### save model and optimizer states
def save_checkpoint(state, is_best, filename='./savemodel/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './savemodel/model_best.pth.tar')
        
### Loads model and optimizer states
def load(net, optimizer, load_best=False):
    if load_best == False:
        checkpoint = torch.load("./savemodel/checkpoint.pth.tar")
    else:
        checkpoint = torch.load("./savemodel/model_best.pth.tar")
    start_epoch = checkpoint['epoch']
    best_pred = checkpoint['best_acc']
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return start_epoch, best_pred
    
resize_images()
dataset, labels = compile_images()

X_train, X_test, y_train, y_test = train_test_split(dataset, labels, \
                                                    test_size = 0.2,\
                                                    random_state = 1,\
                                                    shuffle=True,\
                                                    stratify=labels)
    
transform = transforms.Compose([transforms.ToPILImage(),\
                                transforms.RandomHorizontalFlip(),\
                                transforms.ToTensor(),\
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],\
                                                     std=[0.229, 0.224, 0.225])])
transform_test = transforms.Compose([transforms.ToTensor(),\
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],\
                                                     std=[0.229, 0.224, 0.225])])

trainset = fruits_dataset(dataset=X_train, labels=y_train, transform=transform)
testset = fruits_dataset(dataset=X_test, labels=y_test, transform=transform_test)
train_loader = DataLoader(trainset, batch_size=5,\
                          shuffle=True, num_workers=0, pin_memory=False)
test_loader = DataLoader(testset, batch_size=5,\
                          shuffle=False, num_workers=0, pin_memory=False)

cuda = torch.cuda.is_available()
resnet18 = models.resnet18(pretrained=True)
for i, param in resnet18.named_parameters():
    param.requires_grad = False
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, 3)
if cuda:
    resnet18.cuda()
for name, child in resnet18.named_children():
  for name_2, params in child.named_parameters():
    print(name_2, params.requires_grad)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet18.parameters(), lr=0.005)

losses_per_epoch = []
accuracy_per_epoch = []
start_epoch = 0
epoch_stop = 10
epochs = 50
best_acc = 85
for epoch in range(start_epoch, epochs):
    resnet18.train()
    total_loss = 0.0
    losses_per_batch = []
    for i, data in enumerate(train_loader, 0):
        is_best = False
        inputs, labels = data
        if cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        inputs = inputs.float()
        labels = labels.long()

        optimizer.zero_grad()

        outputs = resnet18(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if i % 10 == 9:    # print every 1000 mini-batches of size = batch_size
            print('[Epoch: %d, %5d/ %d points] total loss per batch: %.3f' %
                  (epoch + 1, (i + 1)*5, len(trainset), total_loss/10))
            losses_per_batch.append(total_loss/10)
            total_loss = 0.0
    losses_per_epoch.append(sum(losses_per_batch)/len(losses_per_batch))
    score = model_eval(resnet18, test_loader, cuda=cuda)
    accuracy_per_epoch.append(score)
    if score > best_acc:
            best_acc = score
            is_best = True
    save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': resnet18.state_dict(),
                'best_acc': score,
                'optimizer' : optimizer.state_dict(),
            }, is_best)
    if epoch == epoch_stop-1:
            break

fig = plt.figure()
ax = fig.add_subplot(222)
ax.scatter([e for e in range(1,epoch_stop+1,1)], losses_per_epoch)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss per batch")
ax.set_title("Loss vs Epoch")
print('Finished Training')

fig2 = plt.figure()
ax2 = fig2.add_subplot(222)
ax2.scatter([e for e in range(1,epoch_stop+1,1)], accuracy_per_epoch)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Accuracy vs Epoch")
print('Finished Training')

'''
transform_test = transforms.Compose([transforms.ToTensor(),\
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],\
                                                     std=[0.229, 0.224, 0.225])])
resnet18 = models.resnet18(pretrained=True)
for i, param in resnet18.named_parameters():
    param.requires_grad = False
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, 3)
checkpoint = torch.load("./savemodel/model_best.pth.tar")
resnet18.load_state_dict(checkpoint['state_dict'])
resnet18.eval()
invlabels_dict = {0:'apple', 1:'orange', 2:'pear'}
img = Image.open(os.path.join(basepath,"orange+fruit9.jpg"))
img = img.resize(size=(224,224))
img = np.array(img)
img = transform_test(img)
output = resnet18(img.reshape(1,3,224,224))
_, predicted = torch.max(output.data, 1)
predicted_class = invlabels_dict[predicted.item()]
print(predicted_class)
'''