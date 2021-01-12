# -*- coding: utf-8 -*-
# @File : csp_darknet.py
# @Author: Runist
# @Time : 2020-12-21 15:27
# @Software: PyCharm
# @Brief: 骨干网络

from tensorflow.keras import layers, regularizers
from nets.DropBlock import DropBlock2D
import tensorflow as tf
import config.config as cfg


class Mish(layers.Layer):
    """
    Mish激活函数
    公式：
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: 任意的. 使用参数 `input_shape`
        - Output: 和输入一样的shape
    Examples:
        >> X_input = layers.Input(input_shape)
        >> X = Mish()(X_input)
    """
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    @staticmethod
    def call(inputs, **kwargs):
        return inputs * tf.tanh(tf.nn.softplus(inputs))

    def get_config(self):
        config = super(Mish, self).get_config()
        return config

    @staticmethod
    def compute_output_shape(input_shape, **kwargs):
        return input_shape


def DarknetConv2D_BN_Mish(inputs, num_filter, kernel_size, strides=(1, 1), bn=True):
    """
    卷积 + 批归一化 + leaky激活，因为大量用到这样的结构，所以这样写
    :param inputs: 输入
    :param num_filter: 卷积个数
    :param kernel_size: 卷积核大小
    :param strides: 步长
    :param bn: 是否使用批归一化
    :return: x
    """
    if strides == (1, 1) or strides == 1:
        padding = 'same'
    else:
        padding = 'valid'

    x = layers.Conv2D(num_filter, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      use_bias=not bn, kernel_regularizer=regularizers.l2(5e-4),  # 只有添加正则化参数，才能调用model.losses方法
                      kernel_initializer=tf.random_normal_initializer(stddev=0.01))(inputs)

    if bn:
        x = layers.BatchNormalization()(x)
        x = Mish()(x)

    return x


def DarknetConv2D_BN_Leaky(inputs, num_filter, kernel_size, strides=(1, 1), bn=True):
    """
    卷积 + 批归一化 + leaky激活，因为大量用到这样的结构，
    其中名字的管理比较麻烦，所以添加了函数内部变量
    :param inputs: 输入
    :param num_filter: 卷积个数
    :param kernel_size: 卷积核大小
    :param strides: 步长
    :param bn: 是否使用批归一化
    :return: x
    """

    if strides == (1, 1) or strides == 1:
        padding = 'same'
    else:
        padding = 'valid'

    x = layers.Conv2D(num_filter, kernel_size=kernel_size,
                      strides=strides, padding=padding,              # 这里的参数是只l2求和之后所乘上的系数
                      use_bias=not bn, kernel_regularizer=regularizers.l2(5e-4),
                      kernel_initializer=tf.random_normal_initializer(stddev=0.01))(inputs)

    if bn:
        x = layers.BatchNormalization()(x)
        # alpha是x < 0时，变量系数
        x = layers.LeakyReLU(alpha=0.1)(x)

    return x


def route_group(input_layer, groups, group_id):
    convs = tf.split(input_layer, num_or_size_splits=groups, axis=-1)
    return convs[group_id]


def tiny_resblock_body(inputs, num_filter):
    """
    残差块
    DarknetConv2D_BN_Leaky + 1次 darknet_block
    :param inputs: 上一层输出
    :param num_filter: conv的卷积核个数，每次残差块是不一样的
    :return: 卷积结果
    """
    # 特征整合
    x = DarknetConv2D_BN_Leaky(inputs, num_filter, 3)
    # 生成一个大的残差边(对应左边的shortcut)
    outer_shortconv = x

    # 通道分割(对应右边的卷积)
    # 这里的分割是把输入进来的特征层的最后一维度(通道数)进行分割，把最后一层分割成等同大小的两部分，取第二部分(为什么只要第二部分呢？)
    x = layers.Lambda(route_group, arguments={'groups': 2, 'group_id': 1})(x)
    x = DarknetConv2D_BN_Leaky(x, num_filter//2, 3)

    # 1次残差块
    # 内部的残差边inner_shortconv
    inner_shortconv = x
    x = DarknetConv2D_BN_Leaky(x, num_filter//2, 3)
    # 堆叠 - 两个特征层通道数都是 num_filter//2，堆叠完之后通道数变成num_filter
    x = layers.Concatenate()([x, inner_shortconv])
    # 进行通道整合 - 将通道数变为num_filter
    x = DarknetConv2D_BN_Leaky(x, num_filter, 1)

    # 第三个tiny_resblock_body会引出来一个有效特征层分支
    feat = x

    # 堆叠 - 两个特征层通道数都是 num_filter，堆叠之后通道数变成2*num_filter
    x = layers.Concatenate()([outer_shortconv, x])
    # 压缩特征层的高和宽
    x = layers.MaxPooling2D(pool_size=[2, 2])(x)

    # 最后对通道数进行整合
    return x, feat


def tiny_darknet_body(inputs):
    """
    tiny_darknet53是yolov4的特征提取网络，输出2个大小的特征层
    :param inputs: 输入图片[n, 416, 416, 3]
    :return:
    """
    # (416, 416, 3) -> (208, 208, 32)
    x = layers.ZeroPadding2D(((1, 0), (1, 0)))(inputs)
    x = DarknetConv2D_BN_Leaky(x, 32, 3, strides=2)

    # (208, 208, 32) -> (104, 104, 64)
    x = layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(x, 64, 3, strides=2)

    # (104, 104, 64) -> (52, 52, 128)
    x, _ = tiny_resblock_body(x, 64)
    # (52, 52, 128) -> (26, 26, 256)
    x, _ = tiny_resblock_body(x, 128)
    # (26, 26, 256) -> (13, 13, 512)
    x, feat26x26 = tiny_resblock_body(x, 256)

    feat13x13 = DarknetConv2D_BN_Leaky(x, 512, 3)

    return feat26x26, feat13x13


