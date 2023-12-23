
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os

img_ttl = "test"
dir_path = f"./{img_ttl}_tgt"

# 图像读取
def image_read(image_path):
    #所有图像已全部在matlab中预处理为灰度图像，故imread参数为0
    img_matrix = cv2.imread(image_path,0)
    return img_matrix


# 创建一个ImageDataGenerator实例并设置数据扩充参数
datagen = ImageDataGenerator(rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')


img_list = os.listdir(dir_path)
for imgname in img_list:
    img_path = dir_path + f"/{imgname}"
    img = load_img(img_path, color_mode = 'grayscale')  # 这是一个PIL图像
    x = img_to_array(img)  # 将其转换为形状 (200, 200) 的numpy数组
    x = x.reshape((1,) + x.shape)  # 将其转换为形状 (1, 200, 200) 的numpy数组
    
    # 下面的.flow()命令生成批量的图像数据并进行扩充
    # 它将无限循环，因此我们需要在某个时刻“打破”这个循环
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir=dir_path,save_prefix=imgname.split('.jpg')[0], save_format='jpg'):
        i += 1
        if i > 30:
            break  # 表示生成30张图片后打破循环