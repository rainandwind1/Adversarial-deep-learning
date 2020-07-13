import torch

img = [i for i in range(784)]
img_tensor = torch.tensor(img, dtype = torch.float32)
img_tensor = img_tensor.reshape([1, 28, 28])
print(img_tensor)