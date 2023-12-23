
import tensorflow as tf
import dataset_create
import model_create
import cv2
import numpy as np


tf.config.experimental_run_functions_eagerly(True)

# 测试集原图像
test_raw_dir_path = "./test_raw"
# 测试集目标图像
test_tgt_dir_path = "./test_tgt"
# 模型文件路径
model_path = "./model2_5"


def load_model(model,model_path):
    latest = tf.train.latest_checkpoint(model_path)
    model.load_weights(latest)


def dncnn(model,model_path,inputs):
    load_model(model,model_path)
    pres = model.predict(inputs)
    return pres


if __name__ == "__main__":
    model = model_create.ResNet(0.35)
    imgs_raw = dataset_create.get_dataset(test_raw_dir_path)
    pres = dncnn(model,model_path,imgs_raw)
    for i in range(pres.shape[0]):
        a = pres[i]
        a = a[:,:,0]
        gray_image = cv2.cvtColor(np.uint8(a * 255), cv2.COLOR_GRAY2BGR)
        cv2.imwrite(f"./res/{i}.jpg", gray_image)
        print("")