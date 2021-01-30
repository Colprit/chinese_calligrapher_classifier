import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv
import csv


transform = tv.transforms.Compose([
    tv.transforms.Grayscale(num_output_channels=1),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5,), (0.5,))
])

data_dir = 'data'

BATCH_SIZE = 8

trainset = tv.datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                   transform)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True,
                                          num_workers=0)

testset = tv.datasets.ImageFolder(os.path.join(data_dir, 'test'),
                                   transform)
testloader = torch.utils.data.DataLoader(testset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True,
                                          num_workers=0)

class_names = trainset.classes

class Net(nn.Module):
    def __init__(self,
        conv1_out_chnls,
        conv1_ker_size,
        pool1_ker_size,
        conv2_out_chnls,
        conv2_ker_size,
        pool2_ker_size,
        fc1_lin,
        fc2_lin
    ):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, conv1_out_chnls, conv1_ker_size)
        self.pool1 = nn.MaxPool2d(pool1_ker_size)
        self.conv2 = nn.Conv2d(conv1_out_chnls, conv2_out_chnls, conv2_ker_size)
        self.pool2 = nn.MaxPool2d(pool2_ker_size)
        self.fc1 = nn.Linear(conv2_out_chnls * pool2_ker_size**2, fc1_lin)
        self.fc2 = nn.Linear(fc1_lin, fc2_lin)
        self.fc3 = nn.Linear(fc2_lin, len(class_names))
    
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 30 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


### PARAMETERS

VERSION = '01'
PATH = f'./chinese_classifier_net_{VERSION}.pth'

TRAIN = False

PARAMS = {
    'conv1_out_chnls' : 12,     # layer 1: convultional
    'conv1_ker_size' : 5,
    'pool1_ker_size' : 3,       # layer 2: max pool
    'conv2_out_chnls' : 30,     # layer 3: convultional
    'conv2_ker_size' : 5,
    'pool2_ker_size' : 4,       # layer 4: max pool
    'fc1_lin' : 256,            # layer 5: linear
    'fc2_lin' : 64              # layer 6: linear
}

# file to record results
results_dir = 'results.csv'
if not os.path.exists(results_dir):
    with open(results_dir, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['version', *PARAMS.keys(), *class_names]
        writer.writerow(header)


net = Net( *PARAMS.values() )

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


if TRAIN:
    train()
    torch.save(net.state_dict(), PATH)
else:
    net = Net(*PARAMS.values())
    net.load_state_dict(torch.load(PATH))
    net.to(device)


# Validate
print('Start validation')

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

accuracies = {name : 100 * class_correct[name] / class_total[name] for name in class_names}

for name, accuracy in accuracies.items():
    print(f'Accuracy of {name} : {accuracy:2.0f}')

# saving results to file
with open(results_dir, 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([VERSION, *PARAMS.values(), *accuracies.values()])
print('Saved to results file')