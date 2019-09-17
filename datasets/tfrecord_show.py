import tensorflow as tf
import numpy as np
import io
from PIL import Image
from jade import *




def tfrecords_show(tfrecord_path):
    dicts, class_names = ReadProTxt("/home/jade/Data/DynamicFreezer/dfsgoods.prototxt")
    with tf.Session() as sess:
        example = tf.train.Example()
        # train_records 表示训练的tfrecords文件的路径
        record_iterator = tf.python_io.tf_record_iterator(path=tfrecord_path)
        for record in record_iterator:
            example.ParseFromString(record)
            f = example.features.feature
            # 解析一个example
            # image_name = f['image/filename'].bytes_list.value[0]
            image_encode = f['image/encoded'].bytes_list.value[0]
            image_height = f['image/height'].int64_list.value[0]
            image_width = f['image/width'].int64_list.value[0]
            text = f['image/class/label'].int64_list.value[0]
            print(text)
            image = io.BytesIO(image_encode)
            image = Image.open(image)
            image = np.asarray(image)
            print(dicts[int(text) + 1]["display_name"])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imshow("result", image)
            cv2.waitKey(0)



if __name__ == '__main__':
    tfrecords_path = "/home/jade/Data/DynamicFreezer/Tfrecords/dfgoods_quilib_2019-05-10_test.tfrecord"
    tfrecords_show(tfrecords_path)
