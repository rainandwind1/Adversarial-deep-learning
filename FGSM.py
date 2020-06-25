import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import torchvision
import copy
from torchvision import transforms
from tqdm import *
import matplotlib.pyplot as plt
from torch.autograd.gradcheck import zero_gradients


# 载入数据集
# 定义数据转换格式
mnist_transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x : x.resize_(28*28))])
# 导入数据，定义数据接口
traindata = torchvision.datasets.MNIST(root="./mnist", train=True, download=True, transform=mnist_transform)
testdata  = torchvision.datasets.MNIST(root="./mnist", train=False, download=True, transform=mnist_transform)
trainloader = torch.utils.data.DataLoader(traindata, batch_size=256, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testdata, batch_size=256, shuffle=True, num_workers=0)



class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(28*28, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        output = self.fc3(x)
        return output

# index = 100
# image = testdata[index][0]
# label = testdata[index][1]
# image.resize_(28,28)
# img = transforms.ToPILImage()(image)
# plt.imshow(img)


# index = 100
# batch = iter(testloader).next() #将testloader转换为迭代器
# image = batch[0][index]
# label = batch[1][index]
# image.resize_(28,28)
# img = transforms.ToPILImage()(image)
# plt.imshow(img)
#
# plt.show()

LOAD_KEY = True
net = Net()
path = './param/net.pkl'
if LOAD_KEY:
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 1e-3, momentum=0.9, weight_decay=1e-04)

if not LOAD_KEY:
    num_epoch = 50
    for epo_i in range(num_epoch):
        losses = 0.0
        for data in trainloader:
            inputs, labels = data
            output = net(inputs)
            loss = loss_fn(output,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.data.item()
        print("Epoch:{}  loss:{}".format(epo_i, losses))
    torch.save(net.state_dict(), path)

correct = 0
total = 0
for data in testloader:
    inputs, labels = data
    output = net(Variable(inputs))
    # print(output.data)
    val, pred = torch.max(output.data, 1)
    total += labels.size(0)
    correct += (pred == labels).sum()
print("预测准确率为：{}/{}".format(correct, total))


index = 101
image = testdata[index][0]
label = testdata[index][1]
output = net(Variable(image))
pred = torch.max(output.data,0)[1]

image.resize_(28, 28)
image = transforms.ToPILImage()(image)
fig1 = plt.figure()
plt.imshow(image)

img = Variable(testdata[index][0].resize_(1, 784), requires_grad=True)
label = torch.tensor([testdata[index][1]])
output = net(img)
loss = loss_fn(output, label)
loss.backward()

epsilon = 0.1
x_grad = torch.sign(img.grad.data)
# print(img.data)
x_adversarial = torch.clamp(img.data + epsilon * x_grad, 0, 1)

out_ad = net(x_adversarial)
pred_ad = torch.max(out_ad.data, 1)[1]
print("原样本预测为：{}  对抗样本预测为：{}".format(torch.max(output, 1)[1], pred_ad))

x_adversarial.resize_(28,28)
img = transforms.ToPILImage()(x_adversarial)
fig2 = plt.figure()
plt.imshow(img)
plt.show()