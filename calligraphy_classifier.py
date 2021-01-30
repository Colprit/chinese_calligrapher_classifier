
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
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool1 = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(4, 4)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc1 = nn.Linear(120, 84)
        self.fc1 = nn.Linear(84, len(class_names))
    
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return x


net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.002, momentum=0.9)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

net.to(device)

def train(num_epochs=2):

    for epoch in range(num_epochs):

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
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0
    
    print('Training: Finished')


train(1)

PATH = './chinese_classifier_net.pth'
torch.save(net.state_dict(), PATH)


# Validate

if(False):
    net = Net()
    net.load_state_dict(torch.load(PATH))
    net.to(device)

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
                label = labels[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

for name in class_names:
    print(f'Accuracy of {name} : {100 * class_correct[name] / class_total[name] : 2d}')