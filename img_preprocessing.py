
# 将输入和输出的所有图片处理至200X200

import numpy as np
import os
import cv2
# 图片文件地址
img_dir_path = "./train_raw"
# 目标分辨率
Target_Height = 200
Target_Width = 200




# 图像读取
def image_read(image_path):
    #所有图像已全部在matlab中预处理为灰度图像，故imread参数为0
    img_matrix = cv2.imread(image_path,0)
    return img_matrix




if __name__ == '__main__':

    img_list = os.listdir(img_dir_path)
    for img in img_list:
        img_path = img_dir_path + f"/{img}"
        img_matrix = image_read(img_path)
        # height = np.size(img_matrix,0)
        # width = np.size(img_matrix,1)
        # #先将矩阵填充为方形，然后缩放
        # if height > width:
        #     img_padded = cv2.copyMakeBorder(img_matrix, 0, 0, int((height-width)/2), int((height-width)/2), cv2.BORDER_CONSTANT, value=(0, 0, 0))
        # else:
        #     img_padded = cv2.copyMakeBorder(img_matrix, int((width-height)/2), int((width-height)/2), 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        # img_resized = cv2.resize(img_padded, (Target_Height,Target_Width))
        # # 预处理只进行一次，直接覆盖原图像
        # cv2.imwrite(img_path,img_resized)

        resized_img = cv2.resize(img_matrix, (200, 200), interpolation = cv2.INTER_AREA)
        
        # 保存新图像
        cv2.imwrite(img_path, resized_img)

    