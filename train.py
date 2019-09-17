#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 作者：2019/8/19 by jade
# 邮箱：jadehh@live.com
# 描述：训练文件
# 最近修改：2019/8/19  上午11:55 modify by jade

from nets import nets_factory
from dataset import dataset_factory
from deployment import model_deploy
from preprocessing import preprocessing_factory
from jade import *
import tensorflow as tf
from tensorflow.contrib import slim

tf.logging.set_verbosity(tf.logging.INFO)
class TrainModel():
    def __init__(self, args):
        self.model_name = args.model_name
        self.save_model_path = args.save_model_path
        self.dataset_name = args.dataset_name
        self.dataset_path = args.dataset_path
        self.checkpoint_path = args.checkpoint_path
        self.image_size = args.image_size
        self.batch_size = args.batch_size
        self.checkpoint_exclude_scopes = args.checkpoint_exclude_scopes

        self.weight_decay = 0.00004
        self.num_epochs_per_decay = 2.0
        self.learning_rate = 0.01
        self.learning_rate_decay_factor = 0.94
        self.train_save_model_path = os.path.join(self.save_model_path,
                                                  self.dataset_name + "_" + self.model_name + "_" +
                                                  "use_checkpoint" + "_" + GetToday()
                                                  )


        CreateSavePath(self.train_save_model_path)
        self.init_deploy_config()
        self.create_global_step()
        self.init_dataset()
        self.init_net()
        self.init_optimizer()






    def configure_learning_rate(self, num_samples_per_epoch, glbal_step):
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

    def configure_optimizer(self, learning_rate):
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

        print("fine-tuning from %s" % checkpoint_path)

        return slim.assign_from_checkpoint_fn(
            checkpoint_path,
            variables_to_restore,
            ignore_missing_vars=False)

    def init_net(self):
        ########################
        #####Select NetWork
        ########################
        self.network_fn = nets_factory.get_network_fn(self.model_name,
                                                      num_classes=self.dataset.num_classes,
                                                      weight_decay=self.weight_decay,
                                                      is_training=True)

        print("init net work")

    def init_dataset(self):
        ########################################################
        ##### 创建一个dataset容器来读取数据集
        #######################################################

        # select the dataset
        print("load dataset",self.dataset_path)
        self.dataset = dataset_factory.get_dataset(
            self.dataset_name, self.dataset_path,
        )

        ############################
        #### 选择图像增强的方式
        ############################

        image_processing_fn = preprocessing_factory.get_preprocessing(
            self.model_name,
            is_training=True,
        )

        with tf.device(self.deploy_config.inputs_device()):
            provider = slim.dataset_data_provider.DatasetDataProvider(
                self.dataset,
                num_readers=4,
                common_queue_capacity=20 * self.batch_size,
                common_queue_min=10 * self.batch_size
            )

            [image, label] = provider.get(['image', 'label'])

            train_image_size = self.image_size or self.network_fn.default_image_size

            image = image_processing_fn(image, train_image_size, train_image_size)

            images, labels = tf.train.batch(
            [image, label],
            batch_size=self.batch_size,
            num_threads=4,
            capacity=5 * self.batch_size
            )
            labels = slim.one_hot_encoding(labels, self.dataset.num_classes)

            self.batch_queue = slim.prefetch_queue.prefetch_queue(
                [images, labels], capacity=2 * self.deploy_config.num_clones
            )
            print("Load dataset")

    def init_deploy_config(self):
        with tf.Graph().as_default():
            self.deploy_config = model_deploy.DeploymentConfig(
                num_clones=1,
                clone_on_cpu=False,
                replica_id=0,
                num_replicas=1,
                num_ps_tasks=0,
            )
        print(" init deploy config")

    def create_global_step(self):
        # create global step
        with tf.device(self.deploy_config.variables_device()):
            self.global_step = slim.create_global_step()
        print("Create global step")

    def init_optimizer(self):
        """
        定义一个优化器
        :return:
        """
        learning_rate = self.configure_learning_rate(self.dataset.num_samples, self.global_step)

        self.optimizer = self.configure_optimizer(learning_rate)
        clones = model_deploy.create_clones(self.deploy_config, self.clone_fn, [self.batch_queue])
        self.total_loss, self.clones_gradients = model_deploy.optimize_clones(
            clones,
            self.optimizer,
            var_list=self.get_variables_to_train())


    def train(self):
        print("Start Training")
        first_clone_scope = self.deploy_config.clone_scope(0)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)
        grad_updates = self.optimizer.apply_gradients(self.clones_gradients,
                                               global_step=self.global_step)
        update_ops.append(grad_updates)

        update_op = tf.group(*update_ops)
        with tf.control_dependencies([update_op]):
            train_tensor = tf.identity(self.total_loss, name='train_op')
        ###########################
        # Kicks off the training. #
        ###########################
        slim.learning.train(
            train_tensor,
            logdir=self.train_save_model_path,
            master="",
            is_chief=True,
            init_fn=self.load_model(),
            startup_delay_steps=GetModelStep(self.train_save_model_path),
            number_of_steps=None,
            log_every_n_steps=1,
            save_summaries_secs=600,
            save_interval_secs=600,
            sync_optimizer=None)


    def clone_fn(self,batch_queue):
      """Allows data parallelism by creating multiple clones of network_fn."""
      images, labels = batch_queue.dequeue()
      #"""Build PNASNet Large model for the ImageNet Dataset."""

      logits, end_points = self.network_fn(images)
      #############################
      # Specify the loss function #
      #############################
      if 'AuxLogits' in end_points:
        slim.losses.softmax_cross_entropy(
            end_points['AuxLogits'], labels,
            label_smoothing=0.0, weights=0.4,
            scope='aux_loss')
      slim.losses.softmax_cross_entropy(
          logits, labels, label_smoothing=0.0, weights=1.0)
      return end_points

    def get_variables_to_train(self):
        """Returns a list of variables to train.

        Returns:
          A list of variables to train by the optimizer.
        """
        if self.checkpoint_exclude_scopes is None:
            return tf.trainable_variables()
        else:
            scopes = [scope.strip() for scope in self.checkpoint_exclude_scopes.split(',')]

        variables_to_train = []
        for scope in scopes:
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            variables_to_train.extend(variables)
        return variables_to_train


if __name__ == '__main__':
    import argparse
    paraser = argparse.ArgumentParser(description="Classify")
    # genearl
    paraser.add_argument("--model_name", default="resnet_v2_101", help="model name")
    paraser.add_argument("--save_model_path",
                         default="/home/jade/Models/Image_Classif/",
                         help="path to save model")
    paraser.add_argument("--dataset_name", default="sdfgoods",
                         help="dataset_name")
    paraser.add_argument("--dataset_path", default=GetRootPath() + "Data/DynamicFreezer/Tfrecords/dfgoods_bigger_2019-05-05_train.tfrecord", help="the dataset_path ")
    paraser.add_argument("--checkpoint_path", default=None, help="the checkpoint_path ")
    paraser.add_argument("--image_size", default=128, help="the image_size")
    paraser.add_argument("--batch_size", default=32, help="the batch_size ")
    paraser.add_argument("--checkpoint_exclude_scopes", default='resnet_v2_101/logits/biases,resnet_v2_101/logits/weights', help="the checkpoint_exclude_scopes ")


    args = paraser.parse_args()
    linear = TrainModel(args)
    linear.train()
