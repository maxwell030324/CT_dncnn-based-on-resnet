file_path = './test_tgt'


import os
import dataset_create
import cv2


dir_list = os.listdir(file_path)

for file_name in dir_list:
        img_matrix = cv2.imread(file_path+"/"+file_name,0)

        cv2.imwrite(file_path+f"/new{dir_list.index(file_name)}.jpg", img_matrix)
        
for file_name in os.listdir(file_path):
    if not file_name.startswith('new'):
        os.remove(file_path+'/'+file_name)
