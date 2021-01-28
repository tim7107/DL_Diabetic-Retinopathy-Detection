####################################Import####################################
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.autograd import Variable
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.utils.data as Data
from PIL import Image 
import pandas as pd
from sklearn.metrics import confusion_matrix

#####################################Training settings#####################################
parser = argparse.ArgumentParser(description='dataset')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 164)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-2)')
parser.add_argument('--momentum', type=float, default=0.1, metavar='M',
                    help='SGD momentum (default: 0.1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
#####################################Start of Def####################################
"""
   Own dataloader
"""
def getData(mode):
    if mode == "train" :
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else :
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)

class RetinopathyLoader(Data.Dataset):
    def __init__(self, root, mode, transform):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        self.transform=transform
        print("> Found %d images..." % (len(self.img_name)))
    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)
    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        img_path = self.root + self.img_name[index] + '.jpeg'
        label = self.label[index]
        img=Image.open(img_path)
        img=self.transform(img)
        return img, label
        
"""
   Resnet18 
"""
class Resnet18(nn.Module):
      def __init__(self):
          super(Resnet18,self).__init__()
          ######Layer 0#####
          self.conv0 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
                nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        )
          ######Layer 1#####
          self.conv1_0 = nn.Sequential (
                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        
          self.conv1_1 = nn.Sequential (
                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
          ######Layer 2#####
          self.conv2_0 = nn.Sequential (
                nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
          self.conv2_ds = nn.Sequential (
                nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False),
                nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
          
          self.conv2_1 = nn.Sequential (
                nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
          ######Layer 3#####
          self.conv3_0 = nn.Sequential (
                nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
          self.conv3_ds = nn.Sequential (
                nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False),
                nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
          
          self.conv3_1 = nn.Sequential (
                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
          ######Layer 4#####
          self.conv4_0 = nn.Sequential (
                nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),  
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
          self.conv4_ds = nn.Sequential (
                nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False),
                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
          
          self.conv4_1 = nn.Sequential (
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.AvgPool2d(kernel_size = 7, stride = 1, padding = 0)
        )
          self.fc1 = nn.Sequential (
                nn.Linear(in_features=512, out_features=5, bias=True)
        ) 
      def forward(self, x):
          ######Layer0######
          x = self.conv0(x)
          
          ######Layer1######
          x = self.conv1_0(x)
          x = self.conv1_1(x)
          temp = x
          
          ######Layer2######
          x = self.conv2_0(x)
          temp = self.conv2_ds(temp)
          x = x + temp
          F.relu(x)
          x = self.conv2_1(x)
          temp = x
          
          ######Layer3######
          x = self.conv3_0(x)
          temp = self.conv3_ds(temp)
          x = x + temp
          F.relu(x)
          x = self.conv3_1(x)
          temp = x
          
          ######Layer4######
          x = self.conv4_0(x)
          temp = self.conv4_ds(temp)
          x = x + temp
          F.relu(x)
          x = self.conv4_1(x)
          
          ######fc######
          x = x.view(x.size(0), -1)
          x = self.fc1(x)
          return x 
      
"""
   learning rate scheduling
"""
def adjust_learning_rate(optimizer, epoch):
    lr=1e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
"""
   training function
"""
def train(epoch,net):
    net.train()
    adjust_learning_rate(optimizer, epoch)
    train_loss=0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.to(device), target.to(device)   
        target = target.to('cuda').long()
        optimizer.zero_grad()
        output = net(data)
        loss = Loss(output, target)

        train_loss += loss.data[0]
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()
        
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))    
    temp=100.*correct.item() / len(train_loader.dataset)
    return temp

"""
   testing function
"""                
def test(epoch,net):
    net.eval()
    test_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        if args.cuda:
            data, target = data.to(device), target.to(device)
        target = target.to('cuda').long()
        with torch.no_grad():
        	output = net(data)
        test_loss += Loss(output, target).data[0]
        #the max log-probability
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct.item() / len(test_loader.dataset)))
    temp=100. * correct.item() / len(test_loader.dataset)
    return temp

"""
   Execution of three activation_function(activation_func)
"""
def execute(net,string):
    train_accuracy=[]
    test_accuracy=[]
    high_acc=0
    for epoch in range(1, args.epochs + 1):
        train_temp=train(epoch,net)
        test_temp=test(epoch,net)
        train_accuracy.append(train_temp)
        test_accuracy.append(test_temp)
        
        final_accuracy=test_accuracy[-1]
        if final_accuracy > high_acc:
            high_acc=final_accuracy
            save(net,string)
        
    #np.save('ResNet18_'+string+'_' +'testingdata',test_accuracy)    
    #np.save('ResNet18_'+string+'_' +'trainingdata',train_accuracy) 
        
    plt.title('Resnet18' ,fontsize=14)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.plot(test_accuracy , label= 'testingdata')
    plt.legend(loc='best')
    plt.plot(train_accuracy, label = 'trainingdata')
    plt.legend(loc='best') 
    return final_accuracy
    
"""
   save
"""
def save(net,string):
  torch.save(net.state_dict(),string + 'Resnet18.pth')

def load(net,string):
  netload= net
  netload.load_state_dict(torch.load(string + 'Resnet18.pth'))
  test(epoch=5 ,net= netload)
  
"""
   Confusion matrix
"""
def confusionmatrix(data,net,string):
    netload =   net                                 
    netload.load_state_dict(torch.load(string + 'Resnet18.pth'))
    netload.eval()
    
    
    labels = [0,1,2,3,4]   
    cmap    = plt.cm.Blues
    for iter_test , (batch_x , batch_y) in enumerate(data):
        if args.cuda:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)    
        gt_x   = batch_x.float()
        gt_y   = batch_y.float().cpu()  
        pred_y = netload(gt_x)
        pred_y = pred_y.cpu()
        pred_y  = np.argmax(pred_y.data.numpy(),axis=1)
        cm = confusion_matrix(gt_y.data.numpy(), pred_y,labels )
        if iter_test == 0:
            final_cm = cm
        else:
            final_cm += cm
    final_cm  = final_cm.astype('float') / final_cm.sum(axis=1)[:, np.newaxis]
    ax = plt.gca()
    im = ax.imshow(final_cm , interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    fmt = '.3f'
    thresh = final_cm.max() / 2.
    for i in range(final_cm .shape[0]):
        for j in range(final_cm .shape[1]):
            ax.text(j, i, format(final_cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if final_cm[i, j] > thresh else "black")
    plt.title('Confusion matrix of ')
    plt.title('Confusion matrix')
    plt.xlabel('Prediction')
    plt.ylabel('Ground Truth')
    plt.show()
#####################################End of Def#####################################
        
        
################Using Class RetinopathyLoader to load data################
"""
   create transformation
"""
train_transform=transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.ToTensor() ])
test_transform=transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.ToTensor() ])                            
                            
train_data = RetinopathyLoader(root = "/home/ubuntu/DL_LAB3/data/", mode = "train", transform = train_transform)
test_data = RetinopathyLoader(root = "/home/ubuntu/DL_LAB3/data/", mode = "test", transform = test_transform)
Loss = nn.CrossEntropyLoss()


################Build dataset structure################
"""
   training dateset 
"""
train_loader=Data.DataLoader(
    dataset=train_data,
    batch_size=4,
    shuffle=True, #randomly choose data
    num_workers=2,
)

"""
   testing dateset 
"""       
test_loader=Data.DataLoader(
    dataset=test_data,
    batch_size=64,
    shuffle=True,
    num_workers=2,
)

################Execution of Resnet18################
"""
   ResNet18
"""
resnet18 = Resnet18()
if args.cuda:
    device = torch.device('cuda')
    resnet18.to(device)
optimizer = optim.SGD(resnet18.parameters(), lr=args.lr)
#execute(net = resnet18,string='original')
#load(net = resnet18, string = 'original')
#plt.show()
#confusionmatrix(data=test_loader,net=resnet18,string='original')

"""
   ResNet18 (pretrained)
"""
pretrain_ResNet18 = models.resnet18(pretrained = True)
if args.cuda:
    device = torch.device('cuda')
    pretrain_ResNet18.to(device)
optimizer = optim.SGD(pretrain_ResNet18.parameters(), lr = args.lr, weight_decay = 5e-4)
#execute(net = pretrain_ResNet18,string = 'pretrain')
load(net = pretrain_ResNet18, string = 'pretrain')
#plt.show()
#confusionmatrix(data = test_loader,net = pretrain_ResNet18,string = 'pretrain')