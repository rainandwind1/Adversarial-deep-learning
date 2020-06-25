import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, datasets
import matplotlib.pyplot as plt
import numpy as np



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


index = 100
ad_example, ad_label = test_data[index], test_label[index]

figure1 = plt.figure()
plt.imshow(ad_example)

img = tf.cast(tf.reshape(ad_example, [-1,28*28]), tf.float32)
img = tf.Variable(img, name = 'ad_example')
with tf.GradientTape() as tape:
    output = net(img)
    ad_label = tf.one_hot(ad_label, depth=10)
    ad_label = tf.constant(ad_label)
    ad_label = tf.expand_dims(ad_label, axis=0)
    loss = losses.categorical_crossentropy(output, ad_label, from_logits = True)
grad = tape.gradient(loss, img)
# print(grad)
delta_ad = tf.sign(grad)
delta_ad = tf.reshape(delta_ad, [28, 28])
ad_exp = tf.clip_by_value(ad_example + epsilon * delta_ad.numpy() * 256, 0, 256)

figure2 = plt.figure()
plt.imshow(ad_exp)

ad_exp = tf.cast(tf.reshape(ad_exp, [-1,28*28]), tf.float32)
output2 = net(ad_exp)
print("原样本预测为：{}， 对抗样本预测为：{}".format(tf.argmax(output, 1).numpy()[0], tf.argmax(output2, 1).numpy()[0]))
plt.show()



