基于残差卷积神经网络的CT影响修复脚本

文件说明：

loss_log：训练结果日志，将loss值变化提取到csv文件中

model2_5：训练后的模型

cal_PSNR.py:根据两个文件夹中的图片信息计算PSNR

csv_plot.py：根据csv文件的数据绘制loss值变化折线图

dataset_create.py：根据文件夹地址，创建其下所有图像构成的四维numpy数组

dncnn_train.ipynb：训练的运行单元

img_expand.py：扩充数据集

img_preprocessing.py：图像预处理

load_dncnn_model.py：调用模型

model_create.py：定义模型结构

noise_adder.m：添加泊松噪声和radon&iradon变换

rgb_to_gray.m：三通道图像转变为单通道图像

tem.py：临时文件，处理各类一次性任务

train_config.py：训练参数的配置模块


使用说明：

在python中打开load_dncnn_model.py，保证“./test_raw”文件夹下有文件，“./res”文件夹存在（可以将代码改得更泛用、更美观，但我懒得改了），直接运行即可在“./res”文件夹中得到修复后的图像
