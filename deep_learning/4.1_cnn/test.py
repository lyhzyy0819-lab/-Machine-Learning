import mpmath
import numpy as np

# stride = 1
#
# rgb_image = np.random.randn(3, 4, 4)
#
# rgb_kernel = np.random.randn(3, 2, 2)
#
# C_in, H, W = rgb_image.shape
# C_kernel, k_h, k_w = rgb_kernel.shape
#
# image_padded = np.pad(rgb_image, pad_width=((0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
#
# _, H_padded, W_padded = image_padded.shape
#
# out_h = (H_padded - k_h) // stride + 1
# out_w = (W_padded - k_w) // stride + 1
#
# output = np.zeros((out_h, out_w))
#
# for i in range(out_h):
#     for j in range(out_w):
#         i_start = i * stride
#         j_start = j * stride
#         receptive_field = image_padded[:, i_start:i_start + k_h, j_start:j_start + k_w]
#         output[i, j] = np.sum(receptive_field * rgb_kernel) + 0.5
# print(output)

'''
 SimpleCNN_Numpy 架构:

  输入 (1, 28, 28)
          │
          ▼
  ┌─────────────────────────┐
  │  Conv1: (8, 1, 3, 3)    │  ← 8个滤波器，每个处理1个通道
  │  He init: sqrt(2/9)     │
  └─────────────────────────┘
          │
          ▼ ReLU → MaxPool
          
  (8, 14, 14)
          │
          ▼
  ┌─────────────────────────┐
  │  Conv2: (16, 8, 3, 3)   │  ← 16个滤波器，每个处理8个通道
  │  He init: sqrt(2/72)    │
  └─────────────────────────┘
          │
          ▼ ReLU → MaxPool → Flatten
          
  (784,)
          │
          ▼
  ┌─────────────────────────┐
  │  FC1: (784, 64)         │
  │  He init: sqrt(2/784)   │
  └─────────────────────────┘
          │
          ▼ ReLU
          
  (64,)
          │
          ▼
  ┌─────────────────────────┐
  │  FC2: (64, 10)          │  ← 10类输出
  │  He init: sqrt(2/64)    │
  └─────────────────────────┘
          │
          ▼
  输出 (10,) → 分类logits
'''


class SimpleCNN_Numpy:
    def __init__(self):
        # 8 个滤波器 1个输入通道 3 * 3核
        self.conv1_w = np.random.randn(8, 1, 3, 3) * np.sqrt(2.0 / (1*3*3))
        self.conv1_b = np.zeros(8)

        # conv2 16个滤波器 8个输入通道啊 3 * 3核
        self.conv2_w = np.random.randn(16, 8, 3, 3) * np.sqrt(2.0 / (8*3*3))
        self.conv2_b = np.zeros(16)

        # FC1
        self.fc1_w = np.random.randn(784, 64) * np.sqrt(2.0 / 784)
        self.fc1_b = np.zeros(64)

        # FC2
        self.fc2_w = np.random.randn(64, 10) * np.sqrt(2.0 / 64)
        self.fc2_b = np.zeros(10)

    def relu(self, x):
        return np.maximum(x, 0)

    def conv2d(self, x, w, b, padding=1):
        C_out, C_in, k_h, k_w = w.shape
        _, H, W = x.shape
        if padding:
            x = np.pad(x, ((0, 0), (padding, padding), (padding, padding)), mode='constant')
        _, H_pad, W_pad = x.shape
        out_h = H_pad - k_h + 1
        out_w = W_pad - k_w + 1

        output = np.zeros((C_out, out_h, out_w))
        for c_out in range(C_out):
            for i in range(out_h):
                for j in range(out_w):
                    window = x[:, i:i+k_h, j:j+k_w]
                    output[c_out, i, j] = np.sum(window * w[c_out]) + b[c_out]
        return output

    def max_pool(self, x, size=2):
        C, H, W = x.shape
        out_h, out_w = H // size, W // size
        output = np.zeros((C, out_h, out_w))

        for c in range(C):
            for i in range(out_h):
                for j in range(out_w):
                    window = x[c, i*size:(i+1)*size, j*size:(j+1)*size]
                    output[c, i, j] = np.max(window)
        return output

    def forward(self, x):
        x = self.conv2d(x, self.conv1_w, self.conv1_b)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.conv2d(x, self.conv2_w, self.conv2_b)
        x = self.relu(x)
        x = self.max_pool(x)

        x  = x.flatten()

        x = x @ self.fc1_w + self.fc1_b
        x = self.relu(x)

        x = x @ self.fc2_w + self.fc2_b

        return x