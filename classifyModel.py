#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：2019/9/17 by jade
# 邮箱：jadehh@live.com
# 描述：分类模型
# 最近修改：2019/9/17  上午10:50 modify by jade
import tensorflow as tf
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.platform import gfile
from dataset import dataset_factory
from nets import nets_factory
import cv2
from preprocessing import preprocessing_factory,inception_preprocessing
import numpy as np
import argparse
from jade import *
from tensorflow.contrib import slim

def write_graph_and_checkpoint(inference_graph_def,
                               input_saver_def,
                               trained_checkpoint_prefix,
                               gpu_memory_fraction=0.8):
  """Writes the graph and the checkpoint into disk."""
  gpu_options1 = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
  sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options1))
  sess.run(tf.global_variables_initializer())
  tf.import_graph_def(inference_graph_def, name='')
  saver = saver_lib.Saver(saver_def=input_saver_def,
                          save_relative_paths=True)
  saver.restore(sess, trained_checkpoint_prefix)
  #[print(n.name) for n in tf.get_default_graph().as_graph_def().node]
  return sess


class ClassifyModel():
    def __init__(self,args):
        self.model_name = args.model_name
        self.model_path = args.model_path
        self.proto_txt_path = args.proto_txt_path
        self.gpu_memory_fraction = args.gpu_memory_fraction
        self.num_classes = args.num_classes
        self.sess,self.probabilities = self.setupModel()
    def setupModel(self):
        with tf.Graph().as_default() as graph:
            image_preprocessing_fn = preprocessing_factory.get_preprocessing(self.model_name,is_training=False)
            network_fn = nets_factory.get_network_fn(
                self.model_name,
                num_classes=(self.num_classes),
                is_training=False)
            image_size = network_fn.default_image_size
            image = tf.placeholder(dtype=tf.uint8,shape=[None,None,None,3],name="input")
            processed_images = tf.map_fn(fn=lambda inp: image_preprocessing_fn(inp, image_size, image_size),
                            elems=image,
                            dtype=tf.float32)
            logits,_ = network_fn(processed_images)
            graph_def = graph.as_graph_def()
            saver = tf.train.Saver(**{})
            input_saver_def = saver.as_saver_def()
            probabilities = tf.nn.softmax(logits)
            sess = write_graph_and_checkpoint(
                inference_graph_def=tf.get_default_graph().as_graph_def(),
                input_saver_def=input_saver_def,
                trained_checkpoint_prefix=GetModelPath(self.model_path),
                gpu_memory_fraction=self.gpu_memory_fraction)
        return sess,probabilities



    def predict(self,img,threshold=0.6):
        if type(img) == str:
            img = cv2.imread(img)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        probabilities = self.sess.run(self.probabilities,feed_dict={'input:0': [img]})
        probabilities = probabilities[0, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x: x[1])]

        categories,class_names = ReadProTxt(self.proto_txt_path)
        chinese_names = []
        english_names = []
        label_ids = []
        scores = []
        for i in range(self.num_classes):
            index = sorted_inds[i]
            chinese_names.append(categories[index+1]["display_name"])
            english_names.append(categories[index+1]["name"])
            label_ids.append(index)
            scores.append(probabilities[index] * 100)
            #print('Probability %0.2f%% => [%s]' % (probabilities[index] * 100, chinese_names[i]),end="")

        return chinese_names[0],english_names[0],label_ids[0],scores[0]





if __name__ == '__main__':
    paraser = argparse.ArgumentParser(description="Detect car")
    # genearl
    paraser.add_argument("--model_name", default="resnet_v2_101", help="model name")
    paraser.add_argument("--model_path",
                             default="/home/jade/Models/Image_Classif/dfgoods_bigger_resnet_v2_101_use_checkpoitns_2019-05-05",
                             help="path to load model")
    paraser.add_argument("--proto_txt_path", default="/home/jade/Data/DynamicFreezer/dfsgoods.prototxt",
                             help="path to labels")
    paraser.add_argument("--num_classes", default=10,
                             help="path to labels")
    paraser.add_argument("--gpu_memory_fraction", default=0.8, help="the memory of gpu")

    args = paraser.parse_args()
    classifyModel = ClassifyModel(args)
    file_list = os.listdir("/home/jade/Data/DynamicFreezer/GoodsClassify_bigger_remove_samll_2019-05-05")
    for file in ["hand"]:
        acc = 0
        num = 0
        root_path = "/home/jade/Data/DynamicFreezer/GoodsClassify_bigger_2019-05-05/"+file
        image_list = GetAllImagesPath(root_path)
        processbar = ProcessBar()
        processbar.count = len(image_list)
        for image_path in image_list:
            processbar.start_time = time.time()
            image = cv2.imread(image_path)
            chinese_name, class_name, label_id, score = classifyModel.predict(image)
            if score > 70:
                save_path = CreateSavePath(os.path.join(root_path,class_name))
                shutil.copy(image_path,os.path.join(save_path,GetLastDir(image_path)))
                os.remove(image_path)
            # cv2.imshow("result", image)
            # cv2.waitKey(0)



