import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from neural_networks.test import X_train


# class Swish(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x):
#         return x * torch.sigmoid(x)
#
# swish = Swish()
# x_test = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
# y_test = swish(x_test)
#
# # print(x_test.numpy())
# # print(y_test.detach().numpy())
#
# model_with_swish = nn.Sequential(
#     nn.Linear(10, 32),
#     Swish(),
#     nn.Linear(32, 16),
#     Swish(),
#     nn.Linear(16, 1),
# )
#
# print(model_with_swish)
#
#
# def swish_tf(x):
#     return x * tf.sigmoid(x)
#
# model_keras = keras.Sequential(
#     layers.Dense(32, input_shape=(10,)),
#     layers.Lambda(swish_tf),
#     layers.Dense(16),
#     layers.Lambda(swish_tf),
#     layers.Dense(1),
# )

# model_keras = keras.Sequential(
#     layers.Dense(32, input_shape=(10,), activation=swish),
#     layers.Dense(16, activation=swish),
#     layers.Dense(1),
# )


# digits = load_digits()
# X, y = digits.data, digits.target
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
#                                                   random_state=42, stratify=y)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)
#
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_val = scaler.transform(X_val)
# X_test = scaler.transform(X_test)
# print(f"训练集: {X_train.shape} 样本")
# print(f"验证集: {X_val.shape} 样本")
# print(f"测试集: {X_test.shape} 样本")
#
# model = keras.Sequential([
#     layers.Dense(128, activation='relu', input_shape=(64, )),
#     layers.Dropout(0.3),
#     layers.Dense(64, activation='relu'),
#     layers.Dropout(0.3),
#     layers.Dense(10, activation='softmax')
# ])
#
# model.compile(
#     optimizer='adam',
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )
#
# print("模型编译完成 ✓")
#
# # 回调函数1: EarlyStopping（早停）
# early_stopping = keras.callbacks.EarlyStopping(
#     monitor='val_loss',
#     patience=10,
#     min_delta=0.001,
#     restore_best_weights=True,
#     verbose=1
# )
# print("✓ EarlyStopping: 验证损失连续10轮不下降时停止")
#
# # 回调函数2: ModelCheckpoint（模型检查点）
# checkpoint_path = 'best_model_exercise1.h5'
# model_checkpoint = keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_path,
#     monitor='val_accuracy',
#     verbose=1,
#     save_best_only=True,
#     save_weights_only=False,
# )
#
# print(f"✓ ModelCheckpoint: 保存最佳模型到 {checkpoint_path}")
#
# # 回调函数3: ReduceLROnPlateau（自动降低学习率）
# reduce_lr = keras.callbacks.ReduceLROnPlateau(
#     monitor='val_loss',
#     factor=0.5,
#     patience=5,
#     min_lr=1e-6,
#     verbose=1,
# )
#
# print("✓ ReduceLROnPlateau: 验证损失停滞5轮时，学习率减半")
#
# callbacks = [reduce_lr, model_checkpoint, early_stopping]
# print("\n【4. 开始训练（带回调函数）】")
#
# history = model.fit(
#     X_train, y_train,
#     epochs=100,
#     batch_size=32,
#     validation_data=(X_val, y_val),
#     callbacks=callbacks,
#     verbose=1
# )
#
# actual_epochs = len(history.history['loss'])
# print(f"实际训练轮数: {actual_epochs} (设置了100，但EarlyStopping提前停止)")
#
# test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
# if os.path.exists(checkpoint_path):
#     best_model = keras.models.load_model(checkpoint_path)
#     best_loss, best_acc = best_model.evaluate(X_test, y_test, verbose=0)
#     print(f"最佳模型准确率: {best_acc:.4f}")
#
#     # 清理临时文件
#     os.remove(checkpoint_path)
#     print(f"\n临时文件 {checkpoint_path} 已清理")



