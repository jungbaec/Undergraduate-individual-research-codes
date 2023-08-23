import math
import numpy as np
import random
import scipy
from numpy.linalg import svd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
# import resource

# Variable Settings
batch_size = 8
hidden1_size = 120
hidden2_size = 40
output_size = 10

train_set = torchvision.datasets.CIFAR10('data.CIFAR10', train = True, download = True, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
test_set = torchvision.datasets.CIFAR10('data.CIFAR10', train = False, download = True, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

# Set learning rate
# lr = lambda epoch: 0.95 ** epoch

'''
# Logdet forward and backward
class LogDet(torch.autograd.Function):
    def forward(ctx, input):
        ctx.save_for_backward(input)
        (sign, logdet) = torch.slogdet(input)
        return logdet

    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output*torch.inverse(input.transpose(0,1))
        return grad_input

# Logdet
Logdet = LogDet()
'''

def Logdet(var):
    (sign, logdet) = torch.slogdet(var)
    return logdet
    
'''
def L1_crit(var):
    return torch.abs(var).sum()
'''
# Using seed
random.seed(400)
torch.manual_seed(400)
# torch.cuda.manual_seed(400)

# Training Settings
train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)

# Test settings
test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = False)

# Design Net
class DPPNet(nn.Module):
    def __init__(self, h1, h2, o):
        super(DPPNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, bias = False)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 8, 5, bias = False)
        self.w1 = nn.Linear(8 * 5 * 5, hidden1_size, bias = False)
        self.w2 = nn.Linear(hidden1_size, hidden2_size, bias = False)
        self.w3 = nn.Linear(hidden2_size, output_size, bias = False)
        
    def forward(self, x):  # feed-forwarding neural net
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 8 * 5 * 5)
        x = torch.relu(self.w1(x))      
        x = torch.relu(self.w2(x))
        x = self.w3(x)
        
        return x

model = DPPNet(h1 = hidden1_size, h2 = hidden2_size, o = output_size)
# model.cuda()
# Constructing optimizer
opt = optim.SGD(params = model.parameters(), lr = 1e-4, momentum = 0.9, weight_decay = 1e-5)
# Constructing scheduler
# scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr)
# Parameters for plot
Acc_not_add_train = []
Acc_not_add_test = []
Acc_add_logdet_w1_train = []
Acc_add_logdet_w1_test = []
Acc_add_logdet_w1_and_w2_train = []
Acc_add_logdet_w1_and_w2_test = []
# Acc_add_logdet_w1_and_w2_and_w3_train = []
# Acc_add_logdet_w1_and_w2_and_w3_test = []

def train(epoch):
    model.train()

    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):

        # Using cuda
        # data, target = data.cuda(), target.cuda()
        
        # Wrap into variable
        data, target = Variable(data), Variable(target)
        
        # parameters: zero gradient
        opt.zero_grad()
        
        # forward, backward pass(back propagation), update weights
        train_outputs = model(data)
        loss = F.cross_entropy(train_outputs, target) # using negative-log-likelihood loss function
        # L2 regularization is already included in SGD optimizer, so I added L1 regularization
        '''
        L1_loss = 0
        for param in model.parameters():
            L1_loss += L1_crit(param)
        
        factor = 1e-3
        loss += factor * L1_loss / len(train_loader.dataset)
        '''
        loss.backward()
        opt.step()
        
        # print statistics
        running_loss += loss.item()
        if batch_idx % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.6f' %(epoch, batch_idx + 1, running_loss / 2000))
            running_loss = 0.0

def test():
    model.eval()
    
    train_loader_f = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = False)
    
    corr_train = 0
    corr_test = 0
    test_loss = 0
    for data, target in train_loader_f:
        # data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        train_outputs = model(data)
        
        pred = torch.max(train_outputs.data, 1)[1] # predict by position of maximum element of the 10x1 output matrix
    
        # count correct prediction
        corr_train += (pred == target).sum().item()
    print('Train accuracy: ', 100. *float(corr_train)/float(len(train_loader.dataset)))
    
    for data, target in test_loader:
        # data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        test_outputs = model(data)
        test_loss += F.cross_entropy(test_outputs, target).item()
    
        # count correct prediction
        pred = torch.max(test_outputs.data, 1)[1] # predict by position of maximum element of the 10x1 output matrix
        corr_test += (pred == target).sum().item()
    print('Test accuracy: ', 100. *float(corr_test)/float(len(test_loader.dataset)))
    print('Test loss: ', test_loss / len(test_loader.dataset))
    Acc_not_add_train.append(100. *float(corr_train)/float(len(train_loader.dataset)))
    Acc_not_add_test.append(100. *float(corr_test)/float(len(test_loader.dataset)))

for epoch in range(1, 26):
    #scheduler.step()
    for g in opt.param_groups:
        print(g['lr'])
    train(epoch)
    test()

# Using seed
random.seed(400)
torch.manual_seed(400)
# torch.cuda.manual_seed(400)

# Training Settings
train_loader_w1 = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)

# Test settings
test_loader_w1 = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = False)

# Design Net
class DPPNet_add_logdet_w1(nn.Module):
    def __init__(self, h1, h2, o):
        super(DPPNet_add_logdet_w1, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, bias = False)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 8, 5, bias = False)
        self.w1 = nn.Linear(8 * 5 * 5, hidden1_size, bias = False)
        self.w2 = nn.Linear(hidden1_size, hidden2_size, bias = False)
        self.w3 = nn.Linear(hidden2_size, output_size, bias = False)
    
    def forward(self, x):  # feed-forwarding neural net
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 8 * 5 * 5)
        x = torch.relu(self.w1(x))      
        x = torch.relu(self.w2(x))
        x = self.w3(x)
        
        return x

model_add_logdet_w1 = DPPNet_add_logdet_w1(h1 = hidden1_size, h2 = hidden2_size, o = output_size)
# model_add_logdet_w1.cuda()
# Constructing optimizer
opt_add_logdet_w1 = optim.SGD(params = model_add_logdet_w1.parameters(), lr = 1e-4, momentum = 0.9, weight_decay = 1e-5)
# Constructing scheduler
#scheduler_add_logdet_w1 = optim.lr_scheduler.LambdaLR(opt_add_logdet_w1, lr_lambda=lr)

def train_add_logdet_w1(epoch):
    model_add_logdet_w1.train()

    running_loss = 0.0
    logdet_w3_av = 0.0
    logdet_w2_w3_av = 0.0
    logdet_w1_w2_w3_av = 0.0
    logdet_conv2_w1_w2_w3_av = 0.0
    logdet_conv1_conv2_w1_w2_w3_av = 0.0

    for batch_idx, (data, target) in enumerate(train_loader_w1):

        # Using cuda
        # data, target = data.cuda(), target.cuda()
        
        # Wrap into variable
        data, target = Variable(data), Variable(target)
        
        # parameters: zero gradient
        opt_add_logdet_w1.zero_grad()
        
        # forward, backward pass(back propagation), update weights
        train_outputs = model_add_logdet_w1(data)
        loss = F.cross_entropy(train_outputs, target) # using negative-log-likelihood loss function
        # L2 regularization is already included in SGD optimizer, so I added L1 regularization
        '''
        L1_loss = 0
        for param in model_add_logdet_h1.parameters():
            L1_loss += L1_crit(param)
        
        factor = 1e-3
        loss += factor * L1_loss / len(train_loader.dataset)
        '''
        index = 0
        Det_loss = 0
        for param in model_add_logdet_w1.parameters():
            
            if index == 0:
                L_conv1 = torch.relu(F.conv2d(param, param)) # same as L_conv1 = torch.relu(F.linear(param.view(6,-1), param.view(6,-1)))

            elif index == 1:
                L_conv2 = torch.relu(F.conv2d(param, param)).view(8,8) 
                L_conv2_ext_p = L_conv2
                for i in range(24):
                    L_conv2_ext_p = torch.cat((L_conv2_ext_p, L_conv2), 0)

                L_conv2_ext = L_conv2_ext_p
                for j in range(24):
                    L_conv2_ext = torch.cat((L_conv2_ext, L_conv2_ext_p), 1)
                
                L_conv1_conv2 = torch.relu(F.conv2d(param, F.conv2d(param, L_conv1))).view(8,8) 
                L_conv1_conv2_ext_p = L_conv1_conv2
                for i in range(24):
                    L_conv1_conv2_ext_p = torch.cat((L_conv1_conv2_ext_p, L_conv1_conv2), 0)

                L_conv1_conv2_ext = L_conv1_conv2_ext_p
                for j in range(24):
                    L_conv1_conv2_ext = torch.cat((L_conv1_conv2_ext, L_conv1_conv2_ext_p), 1)
    
            elif index == 2:
                L_w1 = torch.relu(F.linear(param, param))
                L_conv2_w1 = torch.relu(F.linear(param, F.linear(param, L_conv2_ext)))
                L_conv1_conv2_w1 = torch.relu(F.linear(param, F.linear(param, L_conv1_conv2_ext)))

            elif index == 3:
                L_w2 = torch.relu(F.linear(param, param))
                L_w1_w2 = torch.relu(F.linear(param, F.linear(param, L_w1)))
                L_conv2_w1_w2 = torch.relu(F.linear(param, F.linear(param, L_conv2_w1)))
                L_conv1_conv2_w1_w2 = torch.relu(F.linear(param, F.linear(param, L_conv1_conv2_w1)))

            elif index == 4:
                L_w3 = F.linear(param, param) + 0.01 * torch.eye(output_size)
                logdet_w3 = Logdet(L_w3)
                Det_loss += -(hidden2_size**-2 / (hidden2_size**-2 + hidden1_size**-2 + 200**-2 + (25*150)**-2 + (25*25*75)**-2)) * logdet_w3

                L_w2_w3 = F.linear(param, F.linear(param, L_w2)) + 0.01 * torch.eye(output_size)
                logdet_w2_w3 = Logdet(L_w2_w3)
                Det_loss += -(hidden1_size**-2 / (hidden2_size**-2 + hidden1_size**-2 + 200**-2 + (25*150)**-2 + (25*25*75)**-2)) * logdet_w2_w3
                
                L_w1_w2_w3 = F.linear(param, F.linear(param, L_w1_w2)) + 0.01 * torch.eye(output_size)
                logdet_w1_w2_w3 = Logdet(L_w1_w2_w3)
                Det_loss += -(200**-2 / (hidden2_size**-2 + hidden1_size**-2 + 200**-2 + (25*150)**-2 + (25*25*75)**-2)) * logdet_w1_w2_w3
                
                L_conv2_w1_w2_w3 = F.linear(param, F.linear(param, L_conv2_w1_w2)) + 0.01 * torch.eye(output_size)
                logdet_conv2_w1_w2_w3 = Logdet(L_conv2_w1_w2_w3)
                Det_loss += -((25*150)**-2 / (hidden2_size**-2 + hidden1_size**-2 + 200**-2 + (25*150)**-2 + (25*25*75)**-2)) * logdet_conv2_w1_w2_w3

                L_conv1_conv2_w1_w2_w3 = F.linear(param, F.linear(param, L_conv1_conv2_w1_w2)) + 0.01 * torch.eye(output_size)
                logdet_conv1_conv2_w1_w2_w3 = Logdet(L_conv1_conv2_w1_w2_w3)
                Det_loss += -((25*25*75)**-2 / (hidden2_size**-2 + hidden1_size**-2 + 200**-2 + (25*150)**-2 + (25*25*75)**-2)) * logdet_conv1_conv2_w1_w2_w3
                
            index += 1

        loss += 5e-2 * Det_loss
        loss.backward()
        opt_add_logdet_w1.step()
        
        # print statistics
        running_loss += loss.item()
        logdet_w3_av += logdet_w3.item()
        logdet_w2_w3_av += logdet_w2_w3.item()
        logdet_w1_w2_w3_av += logdet_w1_w2_w3.item()
        logdet_conv2_w1_w2_w3_av += logdet_conv2_w1_w2_w3.item()
        logdet_conv1_conv2_w1_w2_w3_av += logdet_conv1_conv2_w1_w2_w3.item()
        
        if batch_idx % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.6f' %(epoch, batch_idx + 1, running_loss / 2000))
            print(logdet_w3_av / 2000)
            print(logdet_w2_w3_av / 2000)
            print(logdet_w1_w2_w3_av / 2000)
            print(logdet_conv2_w1_w2_w3_av / 2000)
            print(logdet_conv1_conv2_w1_w2_w3_av / 2000)

            running_loss = 0.0
            logdet_w3_av = 0.0
            logdet_w2_w3_av = 0.0
            logdet_w1_w2_w3_av = 0.0
            logdet_conv2_w1_w2_w3_av = 0.0
            logdet_conv1_conv2_w1_w2_w3_av = 0.0

def test_add_logdet_w1():
    model_add_logdet_w1.eval()

    # Training Settings
    train_loader_w1_f = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = False)
  
    corr_train = 0
    corr_test = 0
    test_loss = 0
    for data, target in train_loader_w1_f:
        # data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        train_outputs = model_add_logdet_w1(data)
        
        pred = torch.max(train_outputs.data, 1)[1] # predict by position of maximum element of the 10x1 output matrix
    
        # count correct prediction
        corr_train += (pred == target).sum().item()
    print('Train accuracy: ', 100. *float(corr_train)/float(len(train_loader.dataset)))
    
    for data, target in test_loader_w1:
        # data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        test_outputs = model_add_logdet_w1(data)
        test_loss += F.cross_entropy(test_outputs, target).item()
    
        # count correct prediction
        pred = torch.max(test_outputs.data, 1)[1] # predict by position of maximum element of the 10x1 output matrix
        corr_test += (pred == target).sum().item()
    print('Test accuracy: ', 100. *float(corr_test)/float(len(test_loader.dataset)))
    print('Test loss: ', test_loss / len(test_loader.dataset))
    Acc_add_logdet_w1_train.append(100. *float(corr_train)/float(len(train_loader.dataset)))
    Acc_add_logdet_w1_test.append(100. *float(corr_test)/float(len(test_loader.dataset)))

for epoch in range(1, 26):
    #scheduler_add_logdet_w1.step()
    for g in opt_add_logdet_w1.param_groups:
        print(g['lr'])
    train_add_logdet_w1(epoch)
    test_add_logdet_w1()

plt.figure()
plt.plot(range(1, 26), Acc_not_add_train, label = 'Original')
plt.plot(range(1, 26), Acc_add_logdet_w1_train, label = 'Add conv1, conv2, w1, w2, w3 logdet')
plt.title('Accuracy with respect to logdet factors')
plt.xlabel('training epoch')
plt.ylabel('Train accuracy')
plt.legend()
plt.show()

plt.figure()
plt.plot(range(1, 26), Acc_not_add_test, label = 'Original')
plt.plot(range(1, 26), Acc_add_logdet_w1_test, label = 'Add conv1, conv2, w1, w2, w3 logdet')
plt.title('Accuracy with respect to logdet factors')
plt.xlabel('training epoch')
plt.ylabel('Test accuracy')
plt.legend()
plt.show()

'''
class LogDet(torch.autograd.Function):
    def forward(self, input):
        self.save_for_backward(input)
        return torch.det(Variable(input, requires_grad=True)).data

    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output*torch.inverse(input.transpose(0,1))
        return  grad_input

'''
# print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
'''
pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn

'''
'''
        (sign_w1, logdet_w1) = torch.slogdet(model_add_logdet_w1_and_w2.L_w1)
        # print(logdet_w1)
        loss += -100 * logdet_w1/len(train_loader.dataset)
'''
# corr_train += pred.eq(target.data.view_as(pred)).cpu().sum()
'''
if index == 0:
                # Add log-det term to loss(conv1)
                L_conv1 = torch.einsum('ijkl,jmnl->ikmn', [param.view(6, 3, 1, 25), param.view(6, 3, 1, 25).transpose(0,1)]).view(6, 6) + 0.01 * torch.eye(6)
                logdet_conv1 = Logdet(L_conv1) 
                Det_loss += -(6**2) * logdet_conv1 / (75**2)

            elif index == 2:
                # Add log-det term to loss(conv2)
                L_conv2 = torch.einsum('ijkl,jmnl->ikmn', [param.view(8, 6, 1, 25).transpose(0,1), param.view(8, 6, 1, 25)]).view(6, 6) + 0.01 * torch.eye(6)
                logdet_conv2_t = Logdet(L_conv2)          
                Det_loss += -(6**2) * logdet_conv2_t / (200**2)
                
                # Add log-det term to loss(conv2)
                L_conv2 = torch.einsum('ijkl,jmnl->ikmn', [param.view(8, 6, 1, 25), param.view(8, 6, 1, 25).transpose(0,1)]).view(8, 8) + 0.01 * torch.eye(8)
                logdet_conv2 = Logdet(L_conv2)    
                Det_loss += -(8**2) * logdet_conv2 / (150**2)
                '''
'''
if index == 0:
                # Add log-det term to loss(conv1)
                L_conv1 = F.conv2d(param, param).view(6, 6) + 0.01 * torch.eye(6)
                logdet_conv1 = Logdet(L_conv1) 
                Det_loss += -(6**0.5) * logdet_conv1 / (75**0.5) # 230400 = 3072 * 75, 28224 = 6 * 4704

            elif index == 2:
                # Add log-det term to loss(conv2)
                L_conv2 = F.conv_transpose2d(param.view(8, 150, 1, 1).transpose(0,1), param).view(150, 150) + 0.01 * torch.eye(150)
                logdet_conv2_t = Logdet(L_conv2)          
                Det_loss += -(150**0.5) * logdet_conv2_t / (8**0.5) # 6400 = 8 * 800, 176400 = 150 * 1176
                
                # Add log-det term to loss(conv2)
                L_conv2 = F.conv2d(param, param).view(8, 8) + 0.01 * torch.eye(8)
                logdet_conv2 = Logdet(L_conv2)    
                Det_loss += -(8**0.5) * logdet_conv2 / (150**0.5) # 6400 = 8 * 800, 176400 = 150 * 1176
                '''
'''
if index == 0:
                # Add log-det term to loss(conv1)
                L_conv1 = F.conv2d(param, param).view(6, 6) + 0.01 * torch.eye(6)
                logdet_conv1 = Logdet(L_conv1) 
                Det_loss += -((6**0.5) * (4704**1)) * logdet_conv1 / ((75**0.5) * (3072**1)) # 168 = (6 * 4704)**0.5, 480 = (75 * 3072)**0.5

            elif index == 2:
                # Add log-det term to loss(conv2)
                L_conv2 = F.conv_transpose2d(param.view(8, 150, 1, 1).transpose(0,1), param).view(150, 150) + 0.01 * torch.eye(150)
                logdet_conv2_t = Logdet(L_conv2)          
                Det_loss += -((150**0.5) * (1176**1)) * logdet_conv2_t / ((8**0.5) * (800**1)) # 80 = (8 * 800)**0.5, 420 = (150 * 1176)**0.5
                
                # Add log-det term to loss(conv2)
                L_conv2 = F.conv2d(param, param).view(8, 8) + 0.01 * torch.eye(8)
                logdet_conv2 = Logdet(L_conv2)    
                Det_loss += -((8**0.5) * (800**1)) * logdet_conv2 / ((150**0.5) * (1176**1)) # 80 = (8 * 800)**0.5, 420 = (150 * 1176)**0.5
                '''
# L_conv1 += torch.cat((param.view(1, 6),) * 6, 0)
'''
if index == 0:
                # Add log-det term to loss(conv1)
                L_conv1 = F.conv2d(param, param, model_add_logdet_w1.conv1.bias).view(6, 6) + 0.3 * torch.eye(6)
                logdet_conv1 = Logdet(L_conv1) 
                Det_loss += -(((6**(2/3)) * (4704**(4/3)))/((75**(2/3)) * (3072**(4/3)))) * logdet_conv1 # 168 = (6 * 4704)**0.5, 480 = (75 * 3072)**0.5

            elif index == 2:
                # Add log-det term to loss(conv2)
                L_conv2 = F.conv_transpose2d(param.view(8, 150, 1, 1).transpose(0,1) - torch.cat((model_add_logdet_w1.conv2.bias.view(1, 8),) * 150, 0).view(150, 8, 1, 1), param).view(150, 150) + 0.3 * torch.eye(150)
                logdet_conv2_t = Logdet(L_conv2)
                Det_loss += -(((6**(2/3)) * (4704**(4/3)))/((75**(2/3)) * (3072**(4/3)))) * logdet_conv2_t  # 80 = (8 * 800)**0.5, 420 = (150 * 1176)**0.5
                
                # Add log-det term to loss(conv2)
                L_conv2 = F.conv2d(param, param, model_add_logdet_w1.conv2.bias).view(8, 8) + 0.3 * torch.eye(8)
                logdet_conv2 = Logdet(L_conv2)    
                Det_loss += -(((8**(2/3)) * (800**(4/3)))/((150**(2/3)) * (1176**(4/3)))) * logdet_conv2 # 80 = (8 * 800)**0.5, 420 = (150 * 1176)**0.5
                
            elif index == 4:
                # Add log-det term to loss(w1)
                L_w1 = F.linear(param.transpose(0,1) - torch.cat((model_add_logdet_w1.w1.bias.view(1, hidden1_size),) * 200, 0), param.transpose(0,1)) + 0.3 * torch.eye(200)
                logdet_w1_t = Logdet(L_w1)
                Det_loss += -((200**2)/(120**2)) * logdet_w1_t 
                
                # Add log-det term to loss(w1)
                L_w1 = F.linear(param, param, model_add_logdet_w1.w1.bias) + 0.3 * torch.eye(hidden1_size)
                logdet_w1 = Logdet(L_w1)
                Det_loss += -((120**2)/(200**2)) * logdet_w1 

            elif index == 6:
                # Add log-det term to loss(w2)
                L_w2 = F.linear(param.transpose(0,1) - torch.cat((model_add_logdet_w1.w2.bias.view(1, hidden2_size),) * hidden1_size, 0), param.transpose(0,1)) + 0.3 * torch.eye(hidden1_size)
                logdet_w2_t = Logdet(L_w2)
                Det_loss += -((120**2)/(40**2)) * logdet_w2_t
            
                # Add log-det term to loss(w2)
                L_w2 = F.linear(param, param, model_add_logdet_w1_and_w2.w2.bias) + 0.3 * torch.eye(hidden2_size)
                logdet_w2 = Logdet(L_w2)
                Det_loss += -((40**2)/(120**2)) * logdet_w2
                
            elif index == 8:
                # Add log-det term to loss(w3)
                L_w3 = F.linear(param.transpose(0,1) - torch.cat((model_add_logdet_w1_and_w2.w3.bias.view(1, output_size),) * hidden2_size, 0), param.transpose(0,1)) + 0.3 * torch.eye(hidden2_size)
                logdet_w3_t = Logdet(L_w3)
                Det_loss += -((40**2)/(10**2)) * logdet_w3_t
                '''
'''
if index == 0:
                # Add log-det term to loss(conv1)
                L_conv1 = F.conv2d(param, param).view(6, 6) + 0.3 * torch.eye(6)
                logdet_conv1 = Logdet(L_conv1) 
                Det_loss += (6/75) * torch.trace(L_conv1) - ((6**2)/(75**2)) * logdet_conv1 # 168 = (6 * 4704)**0.5, 480 = (75 * 3072)**0.5

            elif index == 1:
                # Add log-det term to loss(conv2)
                L_conv2 = F.conv_transpose2d(param.view(8, 150, 1, 1).transpose(0,1), param).view(150, 150) + 0.3 * torch.eye(150)
                logdet_conv2_t = Logdet(L_conv2)
                Det_loss += (150/8) * torch.trace(L_conv2) - ((150**2)/(8**2)) * logdet_conv2_t  # 80 = (8 * 800)**0.5, 420 = (150 * 1176)**0.5
                
                # Add log-det term to loss(conv2)
                L_conv2 = F.conv2d(param, param).view(8, 8) + 0.3 * torch.eye(8)
                logdet_conv2 = Logdet(L_conv2)    
                Det_loss += (8/150) * torch.trace(L_conv2) - ((8**2)/(150**2)) * logdet_conv2 # 80 = (8 * 800)**0.5, 420 = (150 * 1176)**0.5
                
            elif index == 2:
                # Add log-det term to loss(w1)
                L_w1 = F.linear(param.transpose(0,1), param.transpose(0,1)) + 0.3 * torch.eye(200)
                logdet_w1_t = Logdet(L_w1)
                Det_loss += (200/120) * torch.trace(L_w1) - ((200**2)/(120**2)) * logdet_w1_t
                
                # Add log-det term to loss(w1)
                L_w1 = F.linear(param, param) + 0.3 * torch.eye(hidden1_size)
                logdet_w1 = Logdet(L_w1)
                Det_loss += (120/200) * torch.trace(L_w1) - ((120**2)/(200**2)) * logdet_w1 

            elif index == 3:
                # Add log-det term to loss(w2)
                L_w2 = F.linear(param.transpose(0,1), param.transpose(0,1)) + 0.3 * torch.eye(hidden1_size)
                logdet_w2_t = Logdet(L_w2)
                Det_loss += (120/40) * torch.trace(L_w2) - ((120**2)/(40**2)) * logdet_w2_t
            
                # Add log-det term to loss(w2)
                L_w2 = F.linear(param, param) + 0.3 * torch.eye(hidden2_size)
                logdet_w2 = Logdet(L_w2)
                Det_loss += (40/120) * torch.trace(L_w2) - ((40**2)/(120**2)) * logdet_w2
                
            elif index == 4:
                # Add log-det term to loss(w3)
                L_w3 = F.linear(param.transpose(0,1), param.transpose(0,1)) + 0.3 * torch.eye(hidden2_size)
                logdet_w3_t = Logdet(L_w3)
                Det_loss += (40/10) * torch.trace(L_w3) - ((40**2)/(10**2)) * logdet_w3_t
                '''
'''
# Using seed
random.seed(400)
torch.manual_seed(400)
# torch.cuda.manual_seed(400)

# Training Settings
train_loader_w1_and_w2 = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)

# Test settings
test_loader_w1_and_w2 = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = False)

# Design Net
class DPPNet_add_logdet_w1_and_w2(nn.Module):
    def __init__(self, h1, h2, o):
        super(DPPNet_add_logdet_w1_and_w2, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, bias = False)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 8, 5, bias = False)
        self.w1 = nn.Linear(8 * 5 * 5, hidden1_size, bias = False)
        self.w2 = nn.Linear(hidden1_size, hidden2_size, bias = False)
        self.w3 = nn.Linear(hidden2_size, output_size, bias = False)
        
    def forward(self, x):  # feed-forwarding neural net
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 8 * 5 * 5)
        x = F.relu(self.w1(x))      
        x = F.relu(self.w2(x))
        x = self.w3(x)
        
        return x

model_add_logdet_w1_and_w2 = DPPNet_add_logdet_w1_and_w2(h1 = hidden1_size, h2 = hidden2_size, o = output_size)
# model_add_logdet_w1_and_w2.cuda()
# Constructing optimizer
opt_add_logdet_w1_and_w2 = optim.SGD(params = model_add_logdet_w1_and_w2.parameters(), lr = 1e-4, momentum = 0.9, weight_decay = 1e-5)
# Constructing scheduler
#scheduler_add_logdet_w1_and_w2 = optim.lr_scheduler.LambdaLR(opt_add_logdet_w1_and_w2, lr_lambda=lr)

def train_add_logdet_w1_and_w2(epoch):
    model_add_logdet_w1_and_w2.train()

    running_loss = 0.0
    logdet_conv2_av = 0.0
    logdet_conv1_conv2_av = 0.0
    logdet_w3_av = 0.0
    logdet_w2_w3_av = 0.0
    logdet_w1_w2_w3_av = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader_w1_and_w2):

        # Using cuda
        # data, target = data.cuda(), target.cuda()
        
        # Wrap into variable
        data, target = Variable(data), Variable(target)
        
        # parameters: zero gradient
        opt_add_logdet_w1_and_w2.zero_grad()
        
        # forward, backward pass(back propagation), update weights
        train_outputs = model_add_logdet_w1_and_w2(data)
        loss = F.cross_entropy(train_outputs, target) # using negative-log-likelihood loss function
        # L2 regularization is already included in SGD optimizer, so I added L1 regularization
        
        L1_loss = 0
        for param in model_add_logdet_h1_and_h2.parameters():
            L1_loss += L1_crit(param)
        
        factor = 1e-3
        loss += factor * L1_loss / len(train_loader.dataset)
        
        index = 0
        Det_loss = 0
        for param in model_add_logdet_w1_and_w2.parameters():

            if index == 0:
                L_conv1 = torch.relu(F.conv2d(param, param)) # same as L_conv1 = torch.relu(F.linear(param.view(6,-1), param.view(6,-1)))

            elif index == 1:
                L_conv2 = torch.relu(F.conv2d(param, param)).view(8,8) + 0.01 * torch.eye(8)
                logdet_conv2 = Logdet(L_conv2)
                Det_loss += -((25*150)**-2 / (hidden2_size**-2 + hidden1_size**-2 + 200**-2 + (25*150)**-2 + (25*25*75)**-2)) * logdet_conv2
                
                L_conv1_conv2 = torch.relu(F.conv2d(param, F.conv2d(param, L_conv1))).view(8,8) + 0.01 * torch.eye(8)
                logdet_conv1_conv2 = Logdet(L_conv1_conv2)
                Det_loss += -((25*25*75)**-2 / (hidden2_size**-2 + hidden1_size**-2 + 200**-2 + (25*150)**-2 + (25*25*75)**-2)) * logdet_conv1_conv2
                
            elif index == 2:
                L_w1 = torch.relu(F.linear(param, param))

            elif index == 3:
                L_w2 = torch.relu(F.linear(param, param))
                L_w1_w2 = torch.relu(F.linear(param, F.linear(param, L_w1)))

            elif index == 4:
                L_w3 = F.linear(param, param) + 0.01 * torch.eye(output_size)
                logdet_w3 = Logdet(L_w3)
                Det_loss += -(hidden2_size**-2 / (hidden2_size**-2 + hidden1_size**-2 + 200**-2 + (25*150)**-2 + (25*25*75)**-2)) * logdet_w3

                L_w2_w3 = F.linear(param, F.linear(param, L_w2)) + 0.01 * torch.eye(output_size)
                logdet_w2_w3 = Logdet(L_w2_w3)
                Det_loss += -(hidden1_size**-2 / (hidden2_size**-2 + hidden1_size**-2 + 200**-2 + (25*150)**-2 + (25*25*75)**-2)) * logdet_w2_w3
                
                L_w1_w2_w3 = F.linear(param, F.linear(param, L_w1_w2)) + 0.01 * torch.eye(output_size)
                logdet_w1_w2_w3 = Logdet(L_w1_w2_w3)
                Det_loss += -(200**-2 / (hidden2_size**-2 + hidden1_size**-2 + 200**-2 + (25*150)**-2 + (25*25*75)**-2)) * logdet_w1_w2_w3
                
                
            index += 1

        loss += 5e-2 * Det_loss 
        loss.backward()
        opt_add_logdet_w1_and_w2.step()

        # print statistics
        running_loss += loss.item()
        logdet_conv2_av += logdet_conv2.item()
        logdet_conv1_conv2_av += logdet_conv1_conv2.item()
        logdet_w3_av += logdet_w3.item()
        logdet_w2_w3_av += logdet_w2_w3.item()
        logdet_w1_w2_w3_av += logdet_w1_w2_w3.item()

        if batch_idx % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.6f' %(epoch, batch_idx + 1, running_loss / 2000))
            print(logdet_conv2_av / 2000)
            print(logdet_conv1_conv2_av / 2000)
            print(logdet_w3_av / 2000)
            print(logdet_w2_w3_av / 2000)
            print(logdet_w1_w2_w3_av / 2000)
            
            logdet_conv2_av = 0.0
            logdet_conv1_conv2_av = 0.0
            logdet_w3_av = 0.0
            logdet_w2_w3_av = 0.0
            logdet_w1_w2_w3_av = 0.0

def test_add_logdet_w1_and_w2():
    model_add_logdet_w1_and_w2.eval()

    # Training Settings
    train_loader_w1_and_w2_f = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = False)
    
    corr_train = 0
    corr_test = 0
    test_loss = 0
    for data, target in train_loader_w1_and_w2_f:
        # data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        train_outputs = model_add_logdet_w1_and_w2(data)
        
        pred = torch.max(train_outputs.data, 1)[1] # predict by position of maximum element of the 10x1 output matrix
    
        # count correct prediction
        corr_train += (pred == target).sum().item()
    print('Train accuracy: ', 100. *float(corr_train)/float(len(train_loader.dataset)))
    
    for data, target in test_loader_w1_and_w2:
        # data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        test_outputs = model_add_logdet_w1_and_w2(data)
        test_loss += F.cross_entropy(test_outputs, target).item()
    
        # count correct prediction
        pred = torch.max(test_outputs.data, 1)[1] # predict by position of maximum element of the 10x1 output matrix
        corr_test += (pred == target).sum().item()
    print('Test accuracy: ', 100. *float(corr_test)/float(len(test_loader.dataset)))
    print('Test loss: ', test_loss / len(test_loader.dataset))
    Acc_add_logdet_w1_and_w2_train.append(100. *float(corr_train)/float(len(train_loader.dataset)))
    Acc_add_logdet_w1_and_w2_test.append(100. *float(corr_test)/float(len(test_loader.dataset)))

for epoch in range(1, 26):
    #scheduler_add_logdet_w1_and_w2.step()
    for g in opt_add_logdet_w1_and_w2.param_groups:
        print(g['lr'])
    train_add_logdet_w1_and_w2(epoch)
    test_add_logdet_w1_and_w2()
'''
