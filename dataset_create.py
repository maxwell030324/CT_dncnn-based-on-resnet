
import os
import cv2
import numpy as np

# 通过文件夹地址，将其下所有图片构成CNN训练集的通用函数

def get_dataset(dir_path):
    img_list = []
    for file in os.listdir(dir_path):
        # 读取灰度图像矩阵，并将其构成列表
        img_matrix = cv2.imread(dir_path+"/"+file,0)
        img_list.append(img_matrix)
    # 将列表中的元素叠放，构成四维张量
    img_tensor = np.stack(img_list, axis=0)
    img_tensor = img_tensor[:,:,:,np.newaxis]
    # 归一化
    img_tensor=img_tensor.astype('float32') / 255.0
    
    return img_tensor







        