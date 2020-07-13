import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, datasets
import matplotlib.pyplot as plt
import numpy as np
import copy

# 绘图中文显示设置
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据加载和预处理
(train_data, train_label), (test_data, test_label) = datasets.mnist.load_data()

# print(train_label[0])

# def preprocess(x, y):
#     x = tf.constant(x, dtype = tf.float32)
#     y = tf.constant(y, dtype = tf.float32)
#     return x, y
def preprocess(x, y): # 自定义的预处理函数
    # 调用此函数时会自动传入x,y 对象，shape 为[b, 28, 28], [b]
    # 标准化到0~1
    x = tf.cast(x, dtype=tf.float32)
    x = tf.reshape(x, [-1, 28*28]) # 打平
    y = tf.cast(y, dtype=tf.int32) # 转成整形张量
    y = tf.one_hot(y, depth=10) # one-hot 编码
    # 返回的x,y 将替换传入的x,y 参数，从而实现数据的预处理功能
    return x,y

train_db = tf.data.Dataset.from_tensor_slices((train_data, train_label))
train_db = train_db.shuffle(10000).batch(256).map(preprocess)
test_db = tf.data.Dataset.from_tensor_slices((test_data, test_label))
test_db = test_db.shuffle(100).batch(256).map(preprocess)
# print(train_db.shape)





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


LOAD_KEY = True
epsilon = 0.1
PATH = './param/FGSM_tf.ckpt'
net = Net()

if LOAD_KEY:
    net.load_weights(PATH)

if not LOAD_KEY:
    optimizer = optimizers.Adam(lr = 1e-3)
    for epo_i in range(30):
        loss_sum = 0.
        for data in train_db:
            with tf.GradientTape() as tape:
                inputs, labels = data
                preds = net(inputs)
                # loss = tf.square(labels-preds)
                loss = losses.categorical_crossentropy(labels, preds, from_logits = True)
                loss = tf.reduce_mean(loss)
                loss_sum += loss
            grads = tape.gradient(loss, net.trainable_variables)
            optimizer.apply_gradients(zip(grads, net.trainable_variables))
        print("Epoch:{}  loss:{}".format(epo_i, loss_sum))
    net.save_weights(PATH)

# test
total_right = 0
for data in test_db:
    inputs, labels = data
    output = net(inputs)
    pred = tf.argmax(output, 1).numpy()
    # print(pred)
    total_right += (pred == tf.argmax(labels, 1).numpy()).sum()

print('准确率为：{}'.format(total_right / test_data.shape[0]))


index = 88
ad_example, ad_label = test_data[index], test_label[index]

figure1 = plt.figure()
plt.imshow(ad_example)


img = tf.cast(tf.reshape(ad_example, [-1,28*28]), tf.float32)
img = tf.Variable(img, name = 'ad_example')


###############################################################
# 按照样本类边界距离排序类结果
f_image = net(img).numpy().flatten()
I = (np.array(f_image)).flatten().argsort()[::-1]
label = I[0]  # 预测结果


input_shape = img.numpy().shape
pert_image = copy.deepcopy(img)
w = np.zeros(input_shape)
r_tot = np.zeros(input_shape)

loop_i = 0
max_iter = 50
overshoot = 0.
with tf.GradientTape(persistent=True) as tape:
    x = tf.Variable(pert_image, name='input_img')
    tape.watch(x)
    fs = net(x)
    fs_list = [fs[0][I[k]] for k in range(len(I))]
    k_i = label


# 迭代移动样本最终产生对抗样本
while k_i == label and loop_i < max_iter:
    pert = np.inf
    grads = tape.gradient(fs_list[0], x)
    orig_grad = grads.numpy().copy()
    for k in range(len(I)):
        grads = tape.gradient(fs_list[k], x)
        cur_grad = grads.numpy().copy()

        # 求样本到类边界距离
        w_k = cur_grad - orig_grad
        f_k = (fs[0][I[k]] - fs[0][I[0]]).numpy()

        pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())

        if pert_k < pert:
            pert = pert_k
            w = w_k

    # 累积移动
    r_i = (pert + 1e-4)*w / np.linalg.norm(w)
    r_tot = np.float32(r_tot + r_i)

    # 计算移动之后的样本，看是否有效, 更新迭代状态变量
    pert_image = img + (1 + overshoot)*tf.constant(r_tot)
    with tf.GradientTape(persistent=True) as tape:
        x = tf.Variable(pert_image)
        fs = net(x)
        fs_list = [fs[0][I[k]] for k in range(len(I))]
        k_i = np.argmax(fs.numpy().flatten())
    loop_i += 1

r_tot = (1 + overshoot) * r_tot

outputs = net(pert_image)
pred = tf.argmax(outputs)
print("原样本预测为：{}  对抗样本预测为：{}".format(I[0], pred.numpy()[0]))

pert_image = tf.reshape(pert_image,(28,28))
fig2 = plt.figure()
plt.title('对抗样本')
plt.imshow(pert_image.numpy())
plt.show()



