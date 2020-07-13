'''
利用前向梯度输出的分量对像素的导数
前向梯度是由神经网络的目标类别输出值对于每一个像素的偏导数组成的。
这意味着，我们可以通过检查每一个像素值对于输出的扰动，选择最合适的像素来进行改变。
1.计算前向导数
2.计算对抗性显著性图
3.添加扰动
'''
import torch
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.autograd.gradcheck import  zero_gradients

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False


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



# 计算前向导数（雅阁比矩阵）
def compute_jacobian(model, inputs):
    output = model(inputs)

    num_features = int(np.prod(inputs.shape[1:]))  # 特征像素点数
    jacobian = torch.zeros([output.size()[1], num_features])
    mask = torch.zeros(output.size())  # 计算微分时使用
    for i in range(output.size()[1]):
        mask[:,i] = 1
        zero_gradients(inputs)
        output.backward(mask, retain_graph=True)
        # 计算输出分量对输入的前向导数
        jacobian[i] = inputs._grad.squeeze().view(-1, num_features).clone()
        mask[:,i] = 0
    return jacobian

def saliency_map(jacobian, target_index, increasing, search_space, nb_features):

    domain = torch.eq(search_space, 1).float()
    all_sum = torch.sum(jacobian, dim=0, keepdim=True) # 按照列求和得到的是输出对每个像素点的总的梯度值
    target_grad = jacobian[target_index] # 得到的是目标输出分量对输入像素点的梯度分量
    others_grad = all_sum - target_grad

    # 剔除不在搜索领域里的像素点
    if increasing:
        increasing_coef = 2  * (torch.eq(domain, 0)).float()
    else:
        increasing_coef = -1 * 2 * (torch.eq(domain, 0)).float()
    increasing_coef = increasing_coef.view(-1, nb_features)

    # 计算任意两个特征的目标前向梯度分量
    target_tmp = target_grad.clone()
    target_tmp -= increasing_coef * torch.max(torch.abs(target_grad))  # 找到最大影响力的输入特征像素点
    alpha = target_tmp.view(-1, 1 ,nb_features) + target_tmp.view(-1, nb_features, 1)

    # 计算除目标类别外的梯度分量之和
    others_tmp = others_grad.clone()
    others_tmp += increasing_coef * torch.max(torch.abs(others_grad))
    beta = others_tmp.view(-1, 1, nb_features) + others_tmp.view(-1, nb_features, 1)

    # alpha, beta 形成一个组合表，用于寻找两个特征对，但是自己和自己的组合需要排除在外
    tmp = np.ones((nb_features, nb_features), int)
    np.fill_diagonal(tmp, 0)
    zero_diagonal = torch.from_numpy(tmp).byte()

    # 更具当前的显著性图来进一步筛选进行修改的特征点
    if increasing:
        mask1 = torch.gt(alpha, 0.0)
        mask2 = torch.lt(beta, 0.0)
    else:
        mask1 = torch.lt(alpha, 0.0)
        mask2 = torch.gt(beta, 0.0)

    # 将mask应用到显著性图上
    mask = torch.mul(torch.mul(mask1, mask2), zero_diagonal.view_as(mask1))
    saliency_map = torch.mul(torch.mul(alpha, torch.abs(beta)), mask.float())
    # 得到最显著的两个像素点（特征对）  注意由于矩阵沿着对角线对称因此会有一对最大值产生
    max_value, max_idx = torch.max(saliency_map.view(-1, nb_features * nb_features), dim=1)
    p = max_idx // nb_features
    q = max_idx % nb_features
    return p, q

def perturbation_single(image, ys_target, theta, gamma, model):
    copy_sample = np.copy(image)
    var_sample = Variable(torch.from_numpy(copy_sample), requires_grad=True)

    outputs = model(var_sample)
    predicted = torch.max(outputs.data, 1)[1]

    var_target = Variable(torch.LongTensor([ys_target, ]))

    if theta > 0:
        increasing = True
    else:
        increasing = False

    num_features = int(np.prod(copy_sample.shape[1:]))
    shape = var_sample.size()

    # 求解最大迭代次数
    max_iters = int(np.ceil(num_features * gamma / 2.0))

    if increasing:
        search_domain = torch.lt(var_sample, 0.99)
    else:
        search_domain = torch.gt(var_sample, 0.01)
    search_domain = search_domain.view(num_features)

    model.eval()
    output = model(var_sample)
    current = torch.max(output.data, 1)[1].numpy()

    iter = 0
    while iter < max_iters and current[0] != ys_target and search_domain.sum() != 0:
        # 计算雅阁比矩阵
        jacobian = compute_jacobian(model, var_sample)
        # 计算最显著的特征点对
        p1, p2 = saliency_map(jacobian, var_target, increasing, search_domain, num_features)
        # 应用
        var_sample_flatten = var_sample.view(-1, num_features)
        var_sample_flatten[0, p1] += theta
        var_sample_flatten[0, p2] += theta

        new_sample = torch.clamp(var_sample_flatten, min=0.0, max=1.0)
        new_sample = new_sample.view(shape)
        search_domain[p1] = 0
        search_domain[p2] = 0
        var_sample = Variable(torch.tensor(new_sample), requires_grad=True)

        output = model(var_sample)
        current = torch.max(output.data, 1)[1].cpu().numpy()
        iter += 1

    adv_samples = var_sample.data.cpu().numpy()
    return  adv_samples

if __name__ == '__main__':
    PATH = './param/net.pkl'

    # 定义数据转换格式
    mnist_transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x:x.resize_(28*28))])
    # 导入数据，定义数据接口
    testdata = torchvision.datasets.MNIST(root = "./mnist", train = False, download = True, transform = mnist_transform)
    testloader = torch.utils.data.DataLoader(testdata, batch_size = 256, shuffle = True, num_workers = 0)

    net = Net()
    checkpoint = torch.load(PATH)
    net.load_state_dict(checkpoint)

    index = 100
    image = testdata[index][0].resize_(1, 784).numpy()
    label = torch.tensor([testdata[index][1]])

    theta = 1.0 # 扰动值
    gamma = 0.1 # 最多扰动特征数占总特征数量的比例
    ys_target = 2 # 目标对抗性样本的标签

    # 生成对抗性样本
    adv_image = perturbation_single(image, ys_target, theta, gamma, net)

    # 绘制图像
    im = adv_image.reshape(28, 28)

    fig1 = plt.figure()
    plt.title("原样本")
    plt.imshow(testdata[index][0].resize_(28, 28).numpy())

    fig2 = plt.figure()
    plt.title("JSMA对抗样本")
    plt.imshow(im)

    # init = torch.max(net(image).data, 1)[0]
    image = torch.tensor(image)
    init = net(image)
    init =  torch.max(init.data, 1)[1]
    adv_image = Variable(torch.from_numpy(adv_image.reshape(1, 784)), requires_grad=True)
    pred = net(adv_image)
    pred = torch.max(pred.data, 1)[1]
    print('原样本预测为：{}  JSMA算法扰动后预测为：{}'.format(init.data[0], pred.numpy()[0]))
    plt.show()





























