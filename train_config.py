
import tensorflow as tf
from tensorflow import keras
import dataset_create
import model_create
import math
import numpy as np
tf.config.experimental_run_functions_eagerly(True)


    # 回调函数
def call_back_only_params(model_path):
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        verbose=0,
        save_weights_only=True,
        save_freq="epoch"
        )
    return ckpt_callback  


# 日志回调函数
def tb_callback(log_path):
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir = log_path,
        histogram_freq=1
        )
    return tensorboard_callback





def load_model(model,model_path):
    latest = tf.train.latest_checkpoint(model_path)
    model.load_weights(latest)


def train(beta,train_dir_name,isreload,reload_path,opt,model_path,log_path,Epochs,Batch_Size,loss_func):
    
    # 获取各个文件夹下图像矩阵构成的张量
    train_raw_np = dataset_create.get_dataset(f"/home/mist/Project/{train_dir_name}_raw")
    train_tgt_np = dataset_create.get_dataset(f"/home/mist/Project/{train_dir_name}_tgt")
    vld_raw_np = dataset_create.get_dataset("/home/mist/Project/vld_raw")
    vld_tgt_np = dataset_create.get_dataset("/home/mist/Project/vld_tgt")

    #train_raw_np = dataset_create.get_dataset(f"./{train_dir_name}_raw")
    #train_tgt_np = dataset_create.get_dataset(f"./{train_dir_name}_tgt")
    #test_raw_np = dataset_create.get_dataset("./test_raw")
    #test_tgt_np = dataset_create.get_dataset("./test_tgt")

    # 实例化模型，beta参数设为beta
    model = model_create.ResNet(beta)
    
    if isreload == True:
        load_model(model,reload_path)
    
    # 设置优化器，指定学习率
    model.compile(optimizer=opt,
              loss=loss_func,
              metrics=['accuracy'])
    
    ckpt_callback = call_back_only_params(model_path)
    tensorboard_callback = tb_callback(log_path)
    model.save_weights(model_path.format(epoch=0))
    
    history = model.fit(
        train_raw_np, 
        train_tgt_np, 
        epochs=Epochs, 
        callbacks=[ckpt_callback,tensorboard_callback],
        verbose=1,
        validation_data=(vld_raw_np, vld_tgt_np),
        batch_size=Batch_Size)
    

    
    
    