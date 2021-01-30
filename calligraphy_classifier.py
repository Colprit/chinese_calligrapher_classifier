import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv


transform = tv.transforms.Compose([
    tv.transforms.Grayscale(num_output_channels=1),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5,), (0.5,))
])

data_dir = 'data'

trainset = tv.datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                   transform)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=8,
                                          shuffle=True,
                                          num_workers=0)

testset = tv.datasets.ImageFolder(os.path.join(data_dir, 'test'),
                                   transform)
testloader = torch.utils.data.DataLoader(testset,
                                          batch_size=8,
                                          shuffle=True,
                                          num_workers=0)

class_names = trainset.classes


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 12, 5)
        self.pool1 = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(12, 30, 5)
        self.pool2 = nn.MaxPool2d(4, 4)
        self.fc1 = nn.Linear(30 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, len(class_names))
    
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 30 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.002, momentum=0.9)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'using: {device}')

net.to(device)

def train(num_epochs=2):

    for epoch in range(num_epochs):
        print(f'Starting epoch: {epoch}')

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 1000 == 999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 1000:.3f}')
                running_loss = 0
    
    print('Training: Finished')

VERSION = '01'
PATH = f'./chinese_classifier_net_{VERSION}.pth'

TRAIN = True
if TRAIN:
    train()
    torch.save(net.state_dict(), PATH)
else:
    net = Net()
    net.load_state_dict(torch.load(PATH))
    net.to(device)

# Validate

class_correct = {name: 0. for name in class_names}
class_total =   {name: 0. for name in class_names}
with torch.no_grad():
    for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[class_names[label]] += correct[i].item()
                class_total[class_names[label]] += 1

for name in class_names:
    print(f'Accuracy of {name} : {100 * class_correct[name] / class_total[name] : 2.0f}')