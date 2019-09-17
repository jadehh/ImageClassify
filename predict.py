#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：2019/9/17 by jade
# 邮箱：jadehh@live.com
# 描述：预测文件
# 最近修改：2019/9/17  上午10:50 modify by jade
import numpy as np
import os
import tensorflow as tf
import matplotlib.pylab as plt
from nets import nets_factory
import cv2



try:
    import urllib2 as urllib
except ImportError:
    import urllib.request as urllib

from datasets import imagenet
from nets import inception
from preprocessing import inception_preprocessing

from tensorflow.contrib import slim

image_size = inception.inception_resnet_v2.default_image_size

with tf.Graph().as_default():

    image_np = cv2.imread("/home/jade/Data/DynamicFreezer/GoodsClassify_04_23/asm-asmnc-pz-yw-500ml/7aadf88a-65a9-11e9-87e4-2cfda1e3a5af.jpg")
    image_np = cv2.cvtColor(image_np,cv2.COLOR_RGB2BGR)
    image = tf.convert_to_tensor(image_np)
    processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
    processed_images = tf.expand_dims(processed_image, 0)

    # Create the model, use the default arg scope to configure the batch norm parameters.
    with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
        logits, _ = inception.inception_resnet_v2(processed_images, num_classes=11, is_training=False)
    probabilities = tf.nn.softmax(logits)

    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join("/home/jade/Models/Image_Classif/dfgoods_inception_resnet_v2_use_checkpoitns_2019-04-29", 'model.ckpt-196478'),
        slim.get_model_variables('InceptionResnetV2'))

    with tf.Session() as sess:
        init_fn(sess)
        np_image, probabilities = sess.run([image, probabilities])
        probabilities = probabilities[0, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x: x[1])]


    names = imagenet.create_readable_names_for_imagenet_labels()
    for i in range(5):
        index = sorted_inds[i]
        print('Probability %0.2f%% => [%s]' % (probabilities[index] * 100, names[index]))

    cv2.imwrite("dog.jpg",cv2.cvtColor(np_image.astype(np.uint8),cv2.COLOR_BGR2RGB))
    plt.figure()
    plt.imshow(np_image.astype(np.uint8))
    plt.axis('off')
    plt.show()