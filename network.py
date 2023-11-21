import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet50, vgg16, VGG16_Weights

class doubleconv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(doubleconv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=7,stride=1,padding=3),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=7,stride=1,padding=3),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class tripleconv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(tripleconv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=7,stride=1,padding=3),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=7,stride=1,padding=3),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=7,stride=1,padding=3),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 7, stride=1, padding=3)
        self.conv12 = nn.Conv2d(64, 64, 7, stride=1, padding=3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 7, stride=1, padding=3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(128, 8, 7, stride=1, padding=3)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(8 * 16 * 16, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.last_fc1 = nn.Linear(512, 3)
        self.last_fc2 = nn.Linear(512, 2)
        self.last_fc3 = nn.Linear(512, 2)
        # self.softmax = nn.Softmax(dim=1)
        
        
        # self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # 64
        x = F.relu(self.conv12(x))
        x = self.pool1(x)
        x = self.pool2(F.relu(self.conv2(x))) # 32
        x = self.pool3(F.relu(self.conv3(x))) # 16
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.dropout(F.relu(self.fc1(x)))
        # print(x.shape)
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(self.fc3(x))
        # TODO: fix last layer activation function
        x = self.last_fc1(x)

        
        return x
    
    
class Net3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 7, stride=1, padding=3)
        self.conv12 = nn.Conv2d(64, 64, 7, stride=1, padding=3)
        # self.pool1 = nn.MaxPool2d(2, 2)     #64
        self.conv2 = nn.Conv2d(64, 128, 7, stride=1, padding=3)
        self.conv22 = nn.Conv2d(128, 128, 7, stride=1, padding=3)
        # self.pool2 = nn.MaxPool2d(2, 2)     #32
        self.conv3 = nn.Conv2d(128, 256, 7, stride=1, padding=3)
        self.conv32 = nn.Conv2d(256, 256, 7, stride=1, padding=3)
        self.conv33 = nn.Conv2d(256, 256, 7, stride=1, padding=3)
        # self.pool3 = nn.MaxPool2d(2, 2)     #16
        self.conv4 = nn.Conv2d(256, 512, 7, stride=1, padding=3)
        self.conv42 = nn.Conv2d(512, 512, 7, stride=1, padding=3)
        self.conv43 = nn.Conv2d(512, 512, 7, stride=1, padding=3)
        # self.pool4 = nn.MaxPool2d(2, 2)     #8
        self.conv5 = nn.Conv2d(512, 512, 7, stride=1, padding=3)
        self.conv52 = nn.Conv2d(512, 512, 7, stride=1, padding=3)
        self.conv53 = nn.Conv2d(512, 512, 7, stride=1, padding=3)
        # self.pool5 = nn.MaxPool2d(2, 2)     #4
        self.fc1 = nn.Linear(512 * 8 * 8, 4096)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 512)
        # self.fc3 = nn.Linear(512, 512)
        self.last_fc1 = nn.Linear(512, 3)
        # self.last_fc2 = nn.Linear(512, 2)
        # self.last_fc3 = nn.Linear(512, 2)
        # self.softmax = nn.Softmax(dim=1)
        
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv12(x))
        x = self.Maxpool(x) # 64
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv22(x))
        x = self.Maxpool(x) # 32
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv32(x))
        x = F.relu(self.conv33(x))
        x = self.Maxpool(x) # 16
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv42(x))
        x = F.relu(self.conv43(x))
        x = self.Maxpool(x) # 8
        # x = F.relu(self.conv5(x))
        # x = F.relu(self.conv52(x))
        # x = F.relu(self.conv53(x))
        # x = self.Maxpool(x) # 4
        
        # x = self.pool1(F.relu(self.conv1(x))) # 64
        # x = self.pool2(F.relu(self.conv2(x))) # 32
        # x = self.pool3(F.relu(self.conv3(x))) # 16
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.dropout(F.relu(self.fc1(x)))
        # print(x.shape)
        x = self.dropout(F.relu(self.fc2(x)))
        # x = self.dropout(self.fc3(x))
        # TODO: fix last layer activation function
        x = self.last_fc1(x)

        
        return x

class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(128, 8, 3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(8 * 16 * 16, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.last_fc1 = nn.Linear(512, 2)
        self.last_fc2 = nn.Linear(512, 2)
        self.last_fc3 = nn.Linear(512, 2)
        # self.softmax = nn.Softmax(dim=1)
        
        
        # self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # print(x.shape)
        x = self.pool(F.relu(self.conv1(x))) # 64
        x = self.pool(F.relu(self.conv2(x))) # 32
        x = self.pool(F.relu(self.conv3(x))) # 16
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.dropout(F.relu(self.fc1(x)))
        # print(x.shape)
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(self.fc3(x))
        # TODO: fix last layer activation function

        x1 = self.last_fc1(x)
        x2 = self.last_fc2(x)
        x3 = self.last_fc3(x)
        # print(x)
        
        return [x1, x2, x3]


# net = Net()

class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4*4*512, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
   
class VGG16v2(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16v2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
            # nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4*4*512, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))
        
        
        self.feature_conv = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5,
            self.layer6,
            self.layer7,
            self.layer8,
            self.layer9,
            self.layer10,
            self.layer11,
            self.layer12,
            self.layer13,
        )
        
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.classifier = nn.Sequential(
            self.fc,
            self.fc1,
            self.fc2
        )
        
        self.gradients = None
        
        
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

        
    def forward(self, x):
        out = self.feature_conv(x)
        out = self.maxpool(out)
        out = out.reshape(out.size(0), -1)
        out = self.classifier(out)
        return out 
    
     # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)
    
    

# reconstruct VGG16, i.e. remove the classifier and replace it with GAP
class VGG(torch.nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.vgg = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)
        self.conv = nn.Sequential(
            self.vgg.features, 
            self.vgg.avgpool 
        ) 
        # self.fc = nn.Linear(512, num_of_class)
        # as we use ImageNet, num_of_class=1000
        self.fc = nn.Linear(512*7*7, 3)

    
    def forward(self,x):    
        x = self.conv(x) # -> (512, 4, 4)
        
        print(x.shape)
        # we use GAP to replace the fc layer, therefore we need to
        # convert (512,7,7) to (512, 7x7)(i.e. each group contains 7x7=49 values), 
        # then convert (512, 7x7) to (512, 1x1) by mean(1)(i.e. average 49 values in each group), 
        # and finally convert (512, 1) to (1, 512) 
        # x = x.view(512,7*7).mean(1).view(1,-1) # -> (1, 512)
        x = x.reshape(x.size(0), -1)
        # FW^T = S
        # where F is the averaged feature maps, which is of shape (1,512)
        # W is the weights for fc layer, which is of shape (1000, 512)
        # S is the scores, which is of shape (1, 1000)
        x = self.fc(x) # -> (1, 1000)
        return x 