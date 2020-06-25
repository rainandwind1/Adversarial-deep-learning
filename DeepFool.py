'''
FGSM算法能够快速简单的生成对抗性样例，但是它没有对原始样本扰动的范围进行界定（扰动程度ϵ是人为指定的），
我们希望通过最小程度的扰动来获得良好性能的对抗性样例，于是DeepFool算法产生
'''
# DeepFool 可以生成模型的最小扰动量的对抗样本，因此可以在这个基础上对不同的模型进行评估
# 本质求样本到分界面的最小距离 之后加上一个微小扰动就可以越界分类错误


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

index = 100
img = Variable(testdata[index][0].resize_(1, 784), requires_grad=True)
label = torch.tensor([testdata[index][1]])

# p =  net(img)
f_image = net(img).data.numpy().flatten()
I = (np.array(f_image)).flatten().argsort()[::-1]
label = I[0]

img1 = testdata[index][0].resize_(28, 28)
img1 = transforms.ToPILImage()(img1)
fig1 = plt.figure()
plt.imshow(img1)

input_shape = img.data.numpy().shape
pert_image = copy.deepcopy(img)
w = np.zeros(input_shape)
r_tot = np.zeros(input_shape)

loop_i = 0
max_iter = 50
overshoot = 0.0
x = Variable(pert_image, requires_grad=True)
fs = net(x)
fs_list = [fs[0][I[k]] for k in range(len(I))]
print(fs_list)
k_i = label

while k_i == label and loop_i < max_iter:
    pert = np.inf
    fs[0][I[0]].backward(retain_graph=True)
    orig_grad = x.grad.data.numpy().copy()
    for k in range(0, len(I)):
        zero_gradients(x)
        fs[0][I[k]].backward(retain_graph=True)
        cur_grad = x.grad.data.numpy().copy()

        w_k = cur_grad - orig_grad
        f_k = (fs[0][I[k]] - fs[0][I[0]]).data.numpy()

        pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())
        if pert_k < pert:
            pert = pert_k
            w = w_k
    # 找到最近的异类
    r_i = (pert + 1e-4) * w / np.linalg.norm(w)
    r_tot = np.float32(r_tot + r_i)

    # 计算移动之后的图像分类，看是否起作用
    pert_image = img + (1 + overshoot)*torch.from_numpy(r_tot)
    x = Variable(pert_image, requires_grad=True)
    fs = net(x)
    k_i = np.argmax(fs.data.numpy().flatten())
    loop_i += 1

r_tot = (1+overshoot) * r_tot

outputs = net(pert_image.data.resize_(1,784))
pred = torch.max(outputs.data,1)[1]
print("原样本预测为：{}  对抗样本预测为：{}".format(I[0], pred.data.numpy()[0]))
# print(pert_image.shape)
pert_image = pert_image.reshape(28, 28)
# print(pert_image.shape)
img = transforms.ToPILImage()(pert_image)
fig2 = plt.figure()
plt.imshow(img)
plt.show()




