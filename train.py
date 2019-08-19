#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：2019/8/19 by jade
# 邮箱：jadehh@live.com
# 描述：TODO
# 最近修改：2019/8/19  上午11:55 modify by jade

from nets import nets_factory
from datasets import dataset_factory
from deployment import model_deploy
from jade import *
import tensorflow as tf
from tensorflow.contrib import slim
def TrainModel():
    def __init__(self,args):
        self.model_name = args.model_name
        self.save_model_path = args.save_model_path
        self.dataset_name = args.dataset_name
        self.dataset_path = args.dataset_path
        self.tfrecord_path = os.path.join(self.dataset_path,"tfrecords")
        self.image_size = args.image_size
        self.batch_size = args.batch_size
        self.checkpoint_exclude_scopes = args.checkpoint_exclude_scopes

        self.num_epochs_per_decay = 2.0
        self.learning_rate = 0.01
        self.learning_rate_decay_factor = 0.94
        self.train_save_model_path = os.path.join(self.save_model_path,
                                                  self.dataset_name + "_" + self.model_name + "_" +
                                                  "use_checkpoint" + "_" + GetToday()
                                                  )

        self.checkpoint_path = "/home/jade/Checkpoints/VGG/vgg_16.ckpt"

        CreateSavePath(self.train_save_model_path)


    def configure_learning_rate(self,num_samples_per_epoch,glbal_step):
        """
        学习率的设置
        :param self:
        :param num_samples_per_epoch:
        :param glbal_step:
        :return:
        """

        decay_steps = int(num_samples_per_epoch * self.num_epochs_per_decay / self.batch_size)
        return tf.train.exponential_decay(
            self.learning_rate,
            glbal_step,
            decay_steps,
            self.learning_rate_decay_factor,
            staircase=True,
            name="exponential_decay_learning_factor"
        )

    def configure_optimizer(self,learning_rate):
        """
        配置优化器
        :param self:
        :param learning_rate:
        :return:
        """
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=0.9,
            momentum=0.9,
            epsilon=1.0
        )
        return optimizer

    def load_model(self):
        """
        加载预训练模型
        :param self:
        :return:
        """
        if self.checkpoint_path is None:
            return None
        if tf.train.latest_checkpoint(self.train_save_model_path):
            assert ValueError("不需要预训练模型了")
            return None
        exclusions = []
        if self.checkpoint_exclude_scopes:
            exclusions = [scope.strip() for scope in self.checkpoint_exclude_scopes.split(",")]

        variables_to_restore = []
        for var in slim.get_model_variables():
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    break
                else:
                    variables_to_restore.append(var)


        if tf.gfile.IsDirectory(self.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_path)
        else:
            checkpoint_path = self.checkpoint_path

        print("fine-tuning from %s"%checkpoint_path)

        return slim.assign_from_checkpoint_fn(
            checkpoint_path,
            variables_to_restore,
            ignore_missing_vars=False)


    def train(self):
        with tf.Graph().as_default():
            deploy_config = model_deploy.DeployedModel(
                num_clones = 1,
                clone_on_cpu = False,
                replica_id = 0,
                num_replicas = 1,
                num_ps_tasks = 0,
            )

            #create global step

            with tf.device(deploy_config.variables_device()):
                global_step = slim.get_global_step()

            #select the dataset
            dataset = dataset_factory.get_dataset(
                self.dataset_name,self.dataset_name,"train",self.tfrecord_path,
            )







