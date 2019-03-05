# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 11:53:26 2018

@author: WT
"""
#import matplotlib
#matplotlib.use('Agg')
from GAN_net import D_net, G_net
import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim
import torch
import shutil

basepath = "./data/_12_Feb19/"

### open train files and compile into df, then save

def resize_images(size=128):
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
    for idx,file in enumerate(os.listdir(basepath)):
        try:
            imagename = os.path.join(basepath,file)
            img = Image.open(imagename)
            img = np.array(img)
            if img.shape == (128, 128, 3):
                dataset.append(img)
        except:
            print(f"Image compile error {idx}")
            continue
    return dataset

class birds_dataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.X = dataset
        self.transform = transform
    def __len__(self):
        return(len(self.X))
    def __getitem__(self,idx):
        img = self.X[idx]
        if self.transform:
            img = self.transform(img)
        return img

### save model and optimizer states
def save_checkpoint(Dstate, Gstate, is_best=False, Dfilename='./savemodel/D_checkpoint.pth.tar',\
                    Gfilename='./savemodel/G_checkpoint.pth.tar'):
    torch.save(Dstate, Dfilename)
    torch.save(Gstate, Gfilename)
    if is_best:
        shutil.copyfile(Dfilename, './savemodel/D_model_best.pth.tar')
        shutil.copyfile(Gfilename, './savemodel/G_model_best.pth.tar')
        
### Loads model and optimizer states
def load(Dnet, Gnet, Doptimizer, Goptimizer, load_best=False):
    if load_best == False:
        Dcheckpoint = torch.load("./savemodel/D_checkpoint.pth.tar")
        Gcheckpoint = torch.load("./savemodel/G_checkpoint.pth.tar")
    else:
        Dcheckpoint = torch.load("./savemodel/D_model_best.pth.tar")
        Gcheckpoint = torch.load("./savemodel/G_model_best.pth.tar")
    start_epoch = Dcheckpoint['epoch']
    Dnet.load_state_dict(Dcheckpoint['state_dict'])
    Doptimizer.load_state_dict(Dcheckpoint['optimizer'])
    Gnet.load_state_dict(Gcheckpoint['state_dict'])
    Goptimizer.load_state_dict(Gcheckpoint['optimizer'])
    return start_epoch

def generator_inputs():
    return lambda batch_size, input_size: torch.rand(batch_size, input_size)

'''
def transform_generated(tensor):
    transform_generated = transforms.Normalize(mean=[0.485, 0.456, 0.406],\
                                                     std=[0.229, 0.224, 0.225])
    for i,_ in enumerate(tensor):
        tensor[i] = tensor[i]/255
        transform_generated(tensor[i])
    return tensor
'''

def decode_generator_image(data,g_input):
    d_img = data; d_img = 255*d_img; d_img = d_img.long().cpu().numpy()
    img = (g_input + 1)*255/2
    img = img.long().cpu().numpy()
    fig2, axes2 = plt.subplots(nrows=1, ncols=2,figsize=(7,7))
    ax2 = axes2.flatten()
    ax2[0].imshow(d_img.transpose(1,2,0))
    ax2[1].imshow(img.transpose(1,2,0))
    return fig2

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)   

#resize_images()
dataset = compile_images()

transform = transforms.Compose([transforms.ToPILImage(),\
                                transforms.RandomHorizontalFlip(),\
                                transforms.ToTensor(),\
                                ])
# transforms.Normalize(mean=[0.485, 0.456, 0.406],\
#                                                     std=[0.229, 0.224, 0.225])

b_size = 5 # batch size
trainset = birds_dataset(dataset=dataset, transform=transform)
train_loader = DataLoader(trainset, batch_size=b_size,\
                          shuffle=True, num_workers=0, pin_memory=False)

cuda = torch.cuda.is_available()
Dnet = D_net()
Gnet = G_net(batch_size=b_size)

'''
## Xavier weights initialization
for p in Dnet.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
for p in Gnet.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
'''
# Normal distribution weights initialization
Dnet.apply(weights_init); Gnet.apply(weights_init)
if cuda:
    Dnet.cuda(); Gnet.cuda()

criterion = nn.BCELoss()
D_optimizer = optim.Adam(Dnet.parameters(), lr=0.0005, betas=(0.5, 0.999))
G_optimizer = optim.Adam(Gnet.parameters(), lr=0.0005, betas=(0.5, 0.999))
#start = load(Dnet, Gnet, D_optimizer, G_optimizer, load_best=False)

Dlosses_per_epoch = []
Glosses_per_epoch = []
G_grad_first = []; G_grad_last = []; D_grad_first = []; D_grad_last = []
start_epoch = 0
epoch_stop = 100
epochs = 2500
best_acc = 85
generator = generator_inputs()
for epoch in range(start_epoch, epochs):
    Dnet.train(); Gnet.train()
    total_dloss = 0.0; total_gloss = 0.0
    Dlosses_per_batch = []; Glosses_per_batch = []; G_grad_first_b = []; G_grad_last_b = []; D_grad_first_b = []; D_grad_last_b = []
    for i, data in enumerate(train_loader, 0):
        is_best = False
        if cuda:
            data = data.cuda()
        data = data.float()

        Dnet.zero_grad()
        
        # train discriminator on real data
        real_outputs = Dnet(data)
        real_loss = criterion(real_outputs, torch.autograd.Variable(torch.ones([real_outputs.size()[0],1])).cuda())
        real_loss.backward()
        # train discriminator on fake data
        fake = Gnet(generator(real_outputs.size()[0],8*8).cuda())
        outputs = (fake.detach() + 1)*255/2
        fake_outputs = Dnet(outputs/255)
        fake_loss = criterion(fake_outputs, torch.autograd.Variable(torch.zeros([fake_outputs.size()[0],1])).cuda())
        fake_loss.backward()
        D_loss = real_loss + fake_loss
        D_grad_first_b.append(Dnet.conv1.weight.grad.mean().item()); D_grad_last_b.append(Dnet.conv5.weight.grad.mean().item())
        # update discriminator's weights
        D_optimizer.step()
        
        # train generator on discriminator's evaluation, to improve generator outputs
        Gnet.zero_grad()
        outputs1 = (fake + 1)*255/2
        d_outputs = Dnet(outputs1/255)
        G_loss = criterion(d_outputs, torch.autograd.Variable(torch.ones([d_outputs.size()[0],1])).cuda()) # G to generate real images
        G_loss.backward()
        G_grad_first_b.append(Gnet.dconv1.weight.grad.mean().item()); G_grad_last_b.append(Gnet.dconv5.weight.grad.mean().item())
        G_optimizer.step()
        
        total_dloss += D_loss.item()
        total_gloss += G_loss.item()
        if i % 10 == 9:    # print every 1000 mini-batches of size = batch_size
            print('[Epoch: %d, %5d/ %d points] D loss per batch: %.5f \t G loss per batch: %.5f' %
                  (epoch + 1, (i + 1)*b_size, len(trainset), total_dloss/10, total_gloss/10))
            print("Gradients D_1 D_5 G_1 G_5: %.5f, %.5f, %.5f, %.5f" % (D_grad_first_b[-1], D_grad_last_b[-1], \
                                                                         G_grad_first_b[-1], G_grad_last_b[-1]))
            Dlosses_per_batch.append(total_dloss/10)
            Glosses_per_batch.append(total_gloss/10)
            total_dloss = 0.0; total_gloss = 0.0; #print(fake[0])
    G_grad_first.append(sum(G_grad_first_b)/len(G_grad_first_b)); G_grad_last.append(sum(G_grad_last_b)/len(G_grad_last_b));
    D_grad_first.append(sum(D_grad_first_b)/len(D_grad_first_b)); D_grad_last.append(sum(D_grad_last_b)/len(D_grad_last_b));
    Dlosses_per_epoch.append(sum(Dlosses_per_batch)/len(Dlosses_per_batch))
    Glosses_per_epoch.append(sum(Glosses_per_batch)/len(Glosses_per_batch))
    
    save_checkpoint(Dstate={'epoch': epoch + 1,
                            'state_dict': Dnet.state_dict(),
                            'optimizer' : D_optimizer.state_dict()},\
                    Gstate={'epoch': epoch + 1,
                            'state_dict': Gnet.state_dict(),
                            'optimizer' : G_optimizer.state_dict()})
    fig2 = decode_generator_image(data[0],fake[0])
    fig2.show()
    plt.savefig(os.path.join("./data/",f"birdy_{epoch}.png"))
    if epoch == epoch_stop-1:
            break

fig = plt.figure()
ax = fig.add_subplot(222)
ax.scatter([e for e in range(1,epoch_stop+1,1)], Dlosses_per_epoch)
ax.set_xlabel("Epoch")
ax.set_ylabel("Discriminator Loss per batch")
ax.set_title("Discriminator Loss vs Epoch")
plt.savefig(os.path.join("./data/",f"DLoss.png"))

fig3 = plt.figure()
ax1 = fig3.add_subplot(222)
ax1.scatter([e for e in range(1,epoch_stop+1,1)], Glosses_per_epoch)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Generator Loss per batch")
ax1.set_title("Generator Loss vs Epoch")
plt.savefig(os.path.join("./data/",f"GLoss.png"))

fig4 = plt.figure()
ax2 = fig4.add_subplot(222)
ax2.scatter([e for e in range(1,epoch_stop+1,1)], D_grad_first)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("D_grad_first per batch")
ax2.set_title("D_grad_first vs Epoch")
plt.savefig(os.path.join("./data/",f"Dgradfirst.png"))

fig5 = plt.figure()
ax3 = fig5.add_subplot(222)
ax3.scatter([e for e in range(1,epoch_stop+1,1)], D_grad_last)
ax3.set_xlabel("Epoch")
ax3.set_ylabel("D_grad_last per batch")
ax3.set_title("D_grad_last vs Epoch")
plt.savefig(os.path.join("./data/",f"Dgradlast.png"))

fig6 = plt.figure()
ax4 = fig6.add_subplot(222)
ax4.scatter([e for e in range(1,epoch_stop+1,1)], G_grad_first)
ax4.set_xlabel("Epoch")
ax4.set_ylabel("G_grad_first per batch")
ax4.set_title("G_grad_first vs Epoch")
plt.savefig(os.path.join("./data/",f"Ggradfirst.png"))

fig7 = plt.figure()
ax5 = fig7.add_subplot(222)
ax5.scatter([e for e in range(1,epoch_stop+1,1)], G_grad_last)
ax5.set_xlabel("Epoch")
ax5.set_ylabel("G_grad_last per batch")
ax5.set_title("G_grad_last vs Epoch")
plt.savefig(os.path.join("./data/",f"Ggradlast.png"))
print('Finished Training')