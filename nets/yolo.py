# -*- coding: utf-8 -*-
# @File : yolo.py
# @Author: Runist
# @Time : 2020-12-21 15:38
# @Software: PyCharm
# @Brief:


from tensorflow.keras import layers, models
import tensorflow as tf
import config.config as cfg
from nets.csp_darknet import tiny_darknet_body, DarknetConv2D_BN_Leaky


def Conv2D_Upsample(inputs, num_filter):
    """
    1次（conv + bn + leaky激活） + 上采样
    :param inputs: 输入层
    :param num_filter: 卷积核个数
    :return: x
    """
    x = DarknetConv2D_BN_Leaky(inputs, num_filter, kernel_size=1)
    x = layers.UpSampling2D(2)(x)

    return x


def yolo_feat_reshape(feat):
    """
    处理一下y_pred的数据，reshape，从b, 13, 13, 75 -> b, 13, 13, 3, 25
    在Keras.model编译前处理是为了loss计算上能匹配
    :param feat:
    :return:
    """
    grid_size = tf.shape(feat)[1]
    reshape_feat = tf.reshape(feat, [-1, grid_size, grid_size, len(cfg.anchor_masks[0]), cfg.num_classes + 5])

    return reshape_feat


def yolo_head(y_pred, anchors, calc_loss=False):
    """
    处理一下y_pred的数据，reshape，从b, 13, 13, 75 -> b, 13, 13, 3, 25
    另外，取名为head是有意义的。因为目标检测大多数分为 - Backbone - Detection head两个部分
    :param y_pred: 预测数据
    :param anchors: 其中一种大小的先验框（总共三种）
    :param calc_loss: 是否计算loss，该函数可以在直接预测的地方用
    :return:
        bbox: 存储了x1, y1 x2, y2的坐标 shape(b, 13, 13 ,3, 4)
        objectness: 该分类的置信度 shape(b, 13, 13 ,3, 1)
        class_probs: 存储了20个分类在sigmoid函数激活后的数值 shape(b, 13, 13 ,3, 20)
        pred_xywh: 把xy(中心点),wh shape(b, 13, 13 ,3, 4)
    """
    grid_size = tf.shape(y_pred)[1]

    # tf.spilt的参数对应：2-(x,y) 2-(w,h) 1-置信度 classes=20-分类数目的得分
    box_xy, box_wh, confidence, class_probs = tf.split(y_pred, (2, 2, 1, cfg.num_classes), axis=-1)
    # 举例：box_xy (13, 13, 3, 2) 3是指三个框，2是xy，其他三个输出类似

    # sigmoid是为了让tx, ty在[0, 1]，防止偏移过多，使得中心点落在一个网络单元格中，这也是激活函数的作用（修正）
    # 而对confidence和class_probs使用sigmoid是为了得到0-1之间的概率
    box_xy = tf.sigmoid(box_xy)
    confidence = tf.sigmoid(confidence)
    class_probs = tf.sigmoid(class_probs)

    # !!! grid[x][y] == (y, x)
    # sigmoid(x) + cx，在这里看，生成grid的原因是要和y_true的格式对齐。
    # 而且加上特征图就是13x13 26x26...一个特征图上的点，就预测一个结果。
    grid_y = tf.tile(tf.reshape(tf.range(grid_size), [-1, 1, 1, 1]), [1, grid_size, 1, 1])
    grid_x = tf.tile(tf.reshape(tf.range(grid_size), [1, -1, 1, 1]), [grid_size, 1, 1, 1])
    grid = tf.concat([grid_x, grid_y], axis=-1)  # [gx, gy, 1, 2]
    grid = tf.cast(grid, tf.float32)

    # 把xy, wh归一化成比例
    # box_xy(b, 13, 13, 3, 2)  grid(13, 13, 1, 2)  grid_size shape-()-13
    # box_wh(b, 13, 13, 3, 2)  anchors_tensor(1, 1, 1, 3, 2)
    box_xy = (box_xy + grid) / tf.cast(grid_size, tf.float32)
    # 要注意，xy除去的是13，wh除去的416，是因为下面wh用的也是416(如果xywh不归一化，和概率值一起训练肯定不收敛啊)
    box_wh = tf.exp(box_wh) * anchors / cfg.input_shape
    # 最后 box_xy、box_wh 都是 (b, 13, 13, 3, 2)

    # 把xy,wh 合并成pred_box在最后一个维度上（axis=-1）
    pred_xywh = tf.concat([box_xy, box_wh], axis=-1)  # original xywh for loss

    if calc_loss:
        return pred_xywh

    return box_xy, box_wh, confidence, class_probs


def tiny_yolo_body():
    height, width = cfg.input_shape
    input_image = layers.Input(shape=(height, width, 3), dtype='float32', name="input_1")  # [b, 416, 416, 3]

    # 生成darknet53的主干模型
    # 首先我们会获取到两个有效特征层,分别是
    # feat1 (26, 26, 256)
    # feat2 (13, 13, 512)
    feat26x26, feat13x13 = tiny_darknet_body(input_image)

    # (13, 13, 512) -> (13, 13, 256)
    y13 = DarknetConv2D_BN_Leaky(feat13x13, 256, kernel_size=1)
    output_13x13 = DarknetConv2D_BN_Leaky(y13, 512, kernel_size=3)

    # FPN特征融合
    output_13x13 = DarknetConv2D_BN_Leaky(output_13x13, len(cfg.anchor_masks[0]) * (cfg.num_classes+5), kernel_size=1, bn=False)

    # Conv2D + UpSampling2D (13, 13, 256) -> (13, 13, 128)
    y13_upsample = Conv2D_Upsample(y13, 128)

    # (26, 26, (128+256))
    y26 = layers.Concatenate()([feat26x26, y13_upsample])

    y26 = DarknetConv2D_BN_Leaky(y26, 256, kernel_size=3)
    output_26x26 = DarknetConv2D_BN_Leaky(y26, len(cfg.anchor_masks[0]) * (cfg.num_classes+5), kernel_size=1, bn=False)

    # 这里output1、output2的shape分别是 26x26, 13x13
    # 然后reshape为 从(b, size, size, 75) -> (b, size, size, 3, 25)
    output_26x26 = layers.Lambda(lambda x: yolo_feat_reshape(x), name='feat26')(output_26x26)
    output_13x13 = layers.Lambda(lambda x: yolo_feat_reshape(x), name='feat13')(output_13x13)

    return models.Model(input_image, [output_13x13, output_26x26])
