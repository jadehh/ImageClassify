#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：2019/9/17 by jade
# 邮箱：jadehh@live.com
# 描述：分类准确率的计算
# 最近修改：2019/9/17  上午10:49 modify by jade
from classifyModel import ClassifyModel
import argparse
from jade import *
import tensorflow as tf
import io


dicts, class_names = ReadProTxt("/home/jade/Data/DynamicFreezer/dfsgoods.prototxt")

with open("/home/jade/Data/StaticDeepFreeze/Tfrecords/labels1.txt",'r') as f:
    results = f.read().split("\n")
class_ids = []
class_names = []
for result in results:
    class_ids.append(result.split(":")[0])
    class_names.append(result.split(":")[1])
def cal(tfrecord_path):
    correct_num = 0
    wrong_num = 0
    num = 0
    with tf.Session() as sess:
        example = tf.train.Example()
        # train_records 表示训练的tfrecords文件的路径
        record_iterator = tf.python_io.tf_record_iterator(path=tfrecord_path)
        for record in record_iterator:
            NoLinePrint("reading tf records %d ..." % (num))
            example.ParseFromString(record)
            f = example.features.feature
            # 解析一个example
            # image_name = f['image/filename'].bytes_list.value[0]
            image_encode = f['image/encoded'].bytes_list.value[0]
            image_height = f['image/height'].int64_list.value[0]
            image_width = f['image/width'].int64_list.value[0]
            text = f['image/class/label'].int64_list.value[0]
            image = io.BytesIO(image_encode)
            image = Image.open(image)
            image = np.asarray(image)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            chines_name,eng_name, labelId,score = classifyModel.predict(image)
            save_path = "/home/jade/Data/DynamicFreezer/GoodsClassify_"+GetToday()

            if eng_name == dicts[int(text) + 1]["name"]:
                #CreateSavePath(os.path.join(save_path, eng_name))
                #save_image_path = os.path.join(save_path, eng_name, str(uuid.uuid1()) + ".jpg")
                #cv2.imwrite(save_image_path, image)
                correct_num = correct_num + 1
                num = num + 1
            else:
                # wrong_num = wrong_num + 1
                # num = num + 1
                if eng_name == "hard":
                    continue
                    #correct_num = correct_num + 1
                else:
                    print(eng_name,dicts[int(text) + 1]["name"])
                    wrong_num = wrong_num + 1
                    num = num + 1
                    cv2.imshow("result",image)
                    cv2.waitKey(0)

                #CreateSavePath(os.path.join(save_path, "hard_sample"))
                #save_image_path = os.path.join(save_path, "hard_sample", str(uuid.uuid1()) + ".jpg")
                #cv2.imwrite(save_image_path, image)
        print(num,correct_num,wrong_num)
        print("acc = %f"%(correct_num / float(num)))
if __name__ == '__main__':
    paraser = argparse.ArgumentParser(description="Classify")
    # genearl
    paraser.add_argument("--model_name", default="resnet_v2_101", help="model name")
    paraser.add_argument("--model_path",
                         default="/home/jade/Models/Image_Classif/dfgoods_quilib_resnet_v2_101_use_checkpoitns_2019-05-10",
                         help="path to load model")
    paraser.add_argument("--proto_txt_path", default="/home/jade/Data/DynamicFreezer/dfsgoods.prototxt",
                         help="path to labels")
    paraser.add_argument("--gpu_memory_fraction", default=0.8, help="the memory of gpu")
    paraser.add_argument("--num_classes", default=10,
                             help="path to labels")
    args = paraser.parse_args()
    classifyModel = ClassifyModel(args)
    cal("/home/jade/Data/DynamicFreezer/Tfrecords/dfgoods_quilib_2019-05-10_train.tfrecord")