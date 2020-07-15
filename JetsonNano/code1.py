# Importing the libraries
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=36,shuffle=True,num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True,transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=36, shuffle=False,num_workers=2)

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')




# Trying out a new network
import torch.nn as nn
import torch.nn.functional as F

class Net2(nn.Module):

    def __init__(self):
        super(Net2,self).__init__()

        self.conv1 = nn.Conv2d(3,5,3)
        self.conv2 = nn.Conv2d(5,10,3)
        self.conv3 = nn.Conv2d(10,20,3)
        self.mp = nn.MaxPool2d(2,stride=2)
        self.linear1 = nn.Linear(2*2*20,200)
        self.linear2 = nn.Linear(200,100)
        self.linear3 = nn.Linear(100,10)
        self.relu = nn.ReLU()
    
    def forward(self,x):

        x = self.mp(self.relu(self.conv1(x)))
        x = self.mp(self.relu(self.conv2(x)))
        x = self.mp(self.relu(self.conv3(x)))
        x = x.view(-1,2*2*20)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)

        return x

net = Net2()

import torch.optim as optim


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.01,momentum=0.9)

# Path for the model
PATH = './cifar_net.pth'
net = Net2()
net.load_state_dict(torch.load(PATH))

dataiter = iter(testloader)
images, labels = dataiter.next()
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(16)))

outputs = net(images)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(16)))


correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))



