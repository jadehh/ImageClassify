import math
import os
import random
import sys
import tensorflow as tf
from datasets import dataset_utils
from jade import *
import argparse





class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg,
                         feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def get_dataset_filename(args, split_name, shard_id):

    output_filename = args.dataset_name + "_"+GetToday()+'_%s.tfrecord' % (
        split_name)
    CreateSavePath(args.output_path)
    return os.path.join(args.output_path, output_filename)



def convert_dataset(split_name, filenames, class_names_to_ids,args):
    """Converts the given filenames to a TFRecord dataset.

      Args:
        split_name: The name of the dataset, either 'train' or 'validation'.
        filenames: A list of absolute paths to png or jpg images.
        class_names_to_ids: A dictionary from class names (strings) to ids
          (integers).
        dataset_dir: The directory where the converted datasets are stored.
      """
    assert split_name in ['train', 'test']

    num_per_shard = int(math.ceil(len(filenames) / float(args.num_shards)))

    with tf.Graph().as_default():
        image_reader = ImageReader()

        with tf.Session('') as sess:
            for shard_id in range(args.num_shards):
                output_filename = get_dataset_filename(args, split_name, shard_id)
                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id + 1) * num_per_shard, len(filenames))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                            i + 1, len(filenames), shard_id))
                        sys.stdout.flush()

                        # Read the filename:
                        image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
                        height, width = image_reader.read_image_dims(sess, image_data)

                        class_name = os.path.basename(os.path.dirname(filenames[i]))
                        class_id = class_names_to_ids[class_name]

                        example = dataset_utils.image_to_tfexample(
                            image_data, b'jpg', height, width, class_id)
                        tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()



def get_filenames_and_classes(dataset_dir, dataset_names,proto_txt_path):
    photo_filenames = []
    _,class_names = ReadProTxt(proto_txt_path)
    for dataset_name in dataset_names:
        root = os.path.join(dataset_dir, dataset_name)
        for filename in os.listdir(root):
            path = os.path.join(root, filename)
            if os.path.isdir(path):
                for filename in os.listdir(path):
                    photo_path = os.path.join(path, filename)
                    photo_filenames.append(photo_path)
            else:
                photo_path = path
                photo_filenames.append(photo_path)
    return photo_filenames,class_names[1:]




def create_tfrecords(args):
    photo_filenames, class_names = get_filenames_and_classes(args.dataset_dir, args.dataset_names,args.proto_txt_path)
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))
    random.seed(args.random_seed)
    random.shuffle(photo_filenames)
    _NUM_VALIDATION = int(args.train_rate * len(photo_filenames))
    training_filenames = photo_filenames[:_NUM_VALIDATION]
    validation_filenames = photo_filenames[_NUM_VALIDATION:]
    print(len(training_filenames),len(validation_filenames))
    convert_dataset('train', training_filenames, class_names_to_ids,
                     args)
    convert_dataset('test', validation_filenames, class_names_to_ids,
                    args)

    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))


if __name__ == '__main__':
    paraser = argparse.ArgumentParser(description="Detect car")
    # genearl
    paraser.add_argument("--dataset_dir", default="/home/jade/Data/DynamicFreezer", help="the dataset dir")
    paraser.add_argument("--dataset_names", default=["GoodsClassify_Equilib_2019-05-10"],
                         help="dataset_name")
    paraser.add_argument("--proto_txt_path", default="/home/jade/Data/DynamicFreezer/dfsgoods.prototxt",
                         help="the proto file path")
    paraser.add_argument("--output_path", default="/home/jade/Data/DynamicFreezer/Tfrecords",
                         help="the output file path")
    paraser.add_argument("--random_seed", default=1,
                         help="方法改变随机数生成器的种子")
    paraser.add_argument("--train_rate", default=0.8,
                         help="随机生成训练文件的比例")
    paraser.add_argument("--num_shards", default=1,
                         help="num of tfrecords")
    paraser.add_argument("--dataset_name", default="dfgoods_quilib",
                         help="num of tfrecords")
    # Seed for repeatability.

    args = paraser.parse_args()
    create_tfrecords(args)