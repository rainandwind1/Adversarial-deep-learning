'''
利用前向梯度输出的分量对像素的导数
前向梯度是由神经网络的目标类别输出值对于每一个像素的偏导数组成的。
这意味着，我们可以通过检查每一个像素值对于输出的扰动，选择最合适的像素来进行改变。
1.计算前向导数
2.计算对抗性显著性图
3.添加扰动
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers, layers, losses, datasets
import numpy as np
import matplotlib.pyplot as plt


tf.compat.v1.enable_eager_execution()


plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class Net(keras.Model):
    def __init__(self):
        super(Net, self).__init__()
        self.net = keras.Sequential([
            layers.Dense(300, activation = tf.nn.relu),
            layers.Dense(100, activation = tf.nn.relu),
            layers.Dense(10)
        ])

    def call(self, inputs):
        output = self.net(inputs)
        return output





PATH = './param/FGSM_tf.ckpt'
net = Net()
net.load_weights(PATH)

(train_data, train_label), (test_data, test_label) = datasets.mnist.load_data()


def preprocess(x, y):
    x = tf.cast(x, dtype = tf.float32)
    x = tf.reshape(x, [-1, 28*28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y

train_db = tf.data.Dataset.from_tensor_slices((train_data, train_label))
train_db.shuffle(10000).batch(256).map(preprocess)

test_db = tf.data.Dataset.from_tensor_slices((test_data, test_label))
test_db.shuffle(10000).batch(256).map(preprocess)

index = 10
init_exp = train_data[index]        # 原始图像
x, label = preprocess(train_data[index], train_label[index])


# print(net(x))
def compute_jacobian(model, inputs):
    op_list = []
    with tf.GradientTape(persistent=True) as tape:
        output = model(inputs)
        for i in range(output.shape[1]):
            op_list.append(output[0][i])

    num_features = int(np.prod(inputs.shape[1:]))
    for i in range(output.shape[1]):
        grads = tape.gradient(op_list[i], inputs)
        # print(grads, grads.shape)
        if i == 0:
            jacobian = tf.reshape(grads, [-1, num_features])
            continue

        cur = tf.reshape(grads, [-1, num_features])
        jacobian = tf.concat([jacobian, cur], axis=0)
    return jacobian


def saliency_map(jacobian, target_index, increasing, search_space, nb_features):

    domain = tf.equal(search_space, 1)
    domain = tf.cast(domain, dtype=tf.float32)
    all_sum = tf.reduce_sum(jacobian, axis=0, keepdims=True)
    target_grad = jacobian[target_index]
    other_grad = all_sum - target_grad

    # 剔除不在搜索领域里的点
    if increasing:
        increasing_coef = 2 * (tf.cast(tf.equal(domain, 0), dtype=tf.float32))
    else:
        increasing_coef = -1 * 2 * (tf.cast(tf.equal(domain, 0), dtype=tf.float32))
    increasing_coef = tf.reshape(increasing_coef, [-1, nb_features])


    # 计算 alpha  beta
    # 计算任意两个特征的目标前向梯度分量
    target_tmp = tf.tile(target_grad, [1 for i in range(len(target_grad.shape))])
    target_tmp -= increasing_coef * tf.maximum(tf.abs(target_grad))
    alpha = tf.reshape(target_tmp, [-1, 1, nb_features]) + tf.reshape(target_tmp, [-1, nb_features, 1])

    # 计算除目标类外的梯度分量之和
    other_tmp = tf.tile(other_grad, [1 for i in range(len(other_grad.shape))])
    other_tmp += increasing_coef * tf.reduce_max(tf.abs(other_tmp))
    beta = tf.reshape(other_tmp,[-1, 1 ,nb_features]) + tf.reshape(other_tmp, [-1, nb_features, 1])

    # alpha  beta组合成一个table  接下来寻找最大的特征对，注意自己和自己的组合需要排除在外
    tmp = np.ones((nb_features, nb_features), int)
    np.fill_diagonal(tmp, 0)
    zero_diagonal = tf.constant(tmp, dtype=tf.int32)

    # 得出显著性图谱
    if increasing:
        mask1 = tf.greater(alpha, 0.0)
        mask2 = tf.less(beta, 0.0)
    else:
        mask1 = tf.less(alpha, 0.0)
        mask2 = tf.greater(beta, 0.0)

    # 将mask应用到显著性图上
    mask = tf.multiply(tf.multiply(mask1, mask2), tf.reshape(zero_diagonal, mask1.shape))
    saliency_map = tf.multiply(tf.multiply(alpha, tf.abs(beta)), tf.cast(mask, dtype = tf.float32))
    max_value = tf.reduce_max(tf.reshape(saliency_map, [-1, nb_features * nb_features]))
    max_idx = tf.argmax(tf.reshape(saliency_map, [-1, nb_features * nb_features]))
    p = max_idx // nb_features
    q = max_idx % nb_features
    return p, q


# 测试计算雅阁比矩阵的函数
# print(compute_jacobian(net, tf.Variable(x)))

def perturbation_single(image, ys_target, theta, gamma, model):
    copy_sample = np.copy(image)
    var_sample = tf.Variable(tf.constant(copy_sample))

    outputs = model(var_sample)
    predicted = tf.argmax(outputs).numpy()[0]

    var_target = tf.Variable(tf.constant([ys_target,], dtype = tf.int64))
    if theta > 0:
        increasing = True
    else:
        increasing = False

    num_features = int(np.prod(copy_sample.shape[1:]))
    shape = var_sample.shape


    # 求解最大迭代次数
    max_iters = int(np.ceil(num_features * gamma / 2.0))

    if increasing:
        search_domain = tf.less(var_sample, 0.99)
    else:
        search_domain = tf.greater(var_sample, 0.01)
    search_domain = tf.reshape(search_domain, [ num_features])

    output = model(var_sample)
    current = tf.argmax(output, 1).numpy()

    iter = 0
    # print(search_domain, tf.cast(search_domain, dtype = tf.int32))
    # print(tf.reduce_sum(int(search_domain)).numpy())
    while iter < max_iters and current != ys_target and tf.reduce_sum( tf.cast(search_domain, dtype = tf.int32)).numpy() != 0:
        jacobian = compute_jacobian(model, var_sample)
        p1, p2 = saliency_map(jacobian, var_target, increasing, search_domain, num_features)
        var_sample_flatten = tf.reshape(var_sample, [-1, num_features])
        var_sample_flatten = var_sample_flatten.numpy()
        var_sample_flatten[0, p1] += theta
        var_sample_flatten[0, p2] += theta
        var_sample_flatten = tf.Variable(tf.constant(var_sample_flatten))

        new_example = tf.clip_by_value(var_sample_flatten, 0.0, 1.0)
        new_example = tf.reshape(new_example, shape)
        search_domain = search_domain.numpy()
        search_domain[p1] = 0
        search_domain[p2] = 0
        var_sample = tf.Variable(tf.constant(new_example))

        output = model(var_sample)
        current = tf.reduce_max(outputs, 1).numpy()
        iter += 1
    adv_samples = var_sample.numpy()
    return adv_samples, current

if __name__ == '__main__':
    PATH = './param/FGSM_tf.ckpt'

    net = Net()
    net.load_weights(PATH)

    index = 10
    image = x.numpy()
    label = train_label[index]

    theta = 1.0 # 扰动值
    gamma = 0.1 # 最多扰动特征数占总特征数量的比例
    ys_target = 2

    adv_image, adv_pred = perturbation_single(image, ys_target, theta, gamma, net)

    im = adv_image.reshape(28, 28)

    fig1 = plt.figure()
    plt.title("原样本")
    plt.imshow(x.numpy().reshape([28, 28]))

    fig2 = plt.figure()
    plt.title("JSMA对抗样本")
    plt.imshow(im)

    init_pred = tf.reduce_max(net(x)).numpy()
    print('原样本预测为：{}  JSMA算法扰动后预测为：{}'.format(init_pred, adv_pred.numpy()))
    plt.show()






