
import tensorflow as tf
from tensorflow.keras import layers

# 密集残差块
class dense_block(tf.keras.layers.Layer):
    def __init__(self,num_channels=64,first_block=False,**kwargs):
        super(dense_block, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(num_channels,
                                   padding='same',
                                   kernel_size=3,
                                   strides=1,
                                   kernel_initializer=tf.keras.initializers.HeNormal())
        self.conv2 = layers.Conv2D(num_channels,
                                   padding='same',
                                   kernel_size=3,
                                   strides=1
                                   )
        self.conv3 = layers.Conv2D(num_channels,
                                   padding='same',
                                   kernel_size=3,
                                   strides=1
                                   )
        self.conv4 = layers.Conv2D(num_channels,
                                   padding='same',
                                   kernel_size=3,
                                   strides=1
                                   )
        self.conv5 = layers.Conv2D(filters=64,
                                   padding='same',
                                   kernel_size=3,
                                   strides=1
                                   )
        self.conv6 = layers.Conv2D(filters=64,
                                   padding='same',
                                   kernel_size=3,
                                   strides=1
                                   )
        self.conv7 = layers.Conv2D(filters=64,
                                   padding='same',
                                   kernel_size=3,
                                   strides=1
                                   )
        self.leaky1 = layers.LeakyReLU(0.01)
        self.leaky2 = layers.LeakyReLU(0.01)
        self.leaky3 = layers.LeakyReLU(0.01)
        self.leaky4 = layers.LeakyReLU(0.01)
        
    def call(self,X):
        X1 = self.leaky1(self.conv1(X)) + X
        X2 = self.leaky2(self.conv2(X1)) + X1 + X
        X3 = self.leaky3(self.conv3(X2)) + X2 + X1 + X
        X4 = self.leaky4(self.conv4(X3)) + X3 + X2 + X1 + X
        return self.conv5(X4)
    

# 残差网络

class ResNet(tf.keras.Model):
    def __init__(self,beta=0.5,**kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.conv=layers.Conv2D(64, kernel_size=(3,3), strides=1, padding='same',input_shape=(None,None,1))
        self.conv_1=layers.Conv2D(64, kernel_size=3, strides=1, padding='same')
        self.conv_2=layers.Conv2D(64, kernel_size=3, strides=1, padding='same')
        self.conv_3=layers.Conv2D(64, kernel_size=3, strides=1, padding='same')
        self.conv_4=layers.Conv2D(64, kernel_size=3, strides=1, padding='same')
        self.conv_5=layers.Conv2D(64, kernel_size=3, strides=1, padding='same')
        self.conv_6=layers.Conv2D(64, kernel_size=3, strides=1, padding='same')
        self.conv_7=layers.Conv2D(64, kernel_size=3, strides=1, padding='same')
        self.conv_8=layers.Conv2D(64, kernel_size=3, strides=1, padding='same')
        self.conv_9=layers.Conv2D(64, kernel_size=3, strides=1, padding='same')
        self.conv_10=layers.Conv2D(64, kernel_size=3, strides=1, padding='same')
        self.conv_11=layers.Conv2D(64, kernel_size=3, strides=1, padding='same')
        self.conv_12=layers.Conv2D(64, kernel_size=3, strides=1, padding='same')
        self.conv_13=layers.Conv2D(64, kernel_size=3, strides=1, padding='same')
        self.conv_14=layers.Conv2D(64, kernel_size=3, strides=1, padding='same')
        self.conv_15=layers.Conv2D(64, kernel_size=3, strides=1, padding='same')
        self.bn_1=layers.BatchNormalization()
        self.bn_2=layers.BatchNormalization()
        self.bn_3=layers.BatchNormalization()
        self.bn_4=layers.BatchNormalization()
        self.bn_5=layers.BatchNormalization()
        self.bn_6=layers.BatchNormalization()
        self.bn_7=layers.BatchNormalization()
        self.bn_8=layers.BatchNormalization()
        self.bn_9=layers.BatchNormalization()
        self.bn_10=layers.BatchNormalization()
        self.bn_11=layers.BatchNormalization()
        self.bn_12=layers.BatchNormalization()
        self.bn_13=layers.BatchNormalization()
        self.bn_14=layers.BatchNormalization()
        self.bn_15=layers.BatchNormalization()
        self.leaky = layers.LeakyReLU(0.01)
        self.leaky_1 = layers.LeakyReLU(0.01)
        self.leaky_2 = layers.LeakyReLU(0.01)
        self.leaky_3 = layers.LeakyReLU(0.01)
        self.leaky_4 = layers.LeakyReLU(0.01)
        self.leaky_5 = layers.LeakyReLU(0.01)
        self.leaky_6 = layers.LeakyReLU(0.01)
        self.leaky_7 = layers.LeakyReLU(0.01)
        self.leaky_8 = layers.LeakyReLU(0.01)
        self.leaky_9 = layers.LeakyReLU(0.01)
        self.leaky_10 = layers.LeakyReLU(0.01)
        self.leaky_11 = layers.LeakyReLU(0.01)
        self.leaky_12 = layers.LeakyReLU(0.01)
        self.leaky_13 = layers.LeakyReLU(0.01)
        self.leaky_14 = layers.LeakyReLU(0.01)
        self.leaky_15 = layers.LeakyReLU(0.01)
        self.dense_block1=dense_block()
        self.dense_block2=dense_block()
        self.dense_block3=dense_block()
        self.dense_block4=dense_block()
        self.dense_block5=dense_block()
        self.lst=layers.Conv2D(1, kernel_size=3, strides=1, padding='same',kernel_regularizer=tf.keras.regularizers.l1(0.01))
        self.beta=beta

    def call(self, x):
        y=x
        x=self.conv(x)
        x1=self.leaky(x)
        x2=self.dense_block1(x1)*self.beta+x1
        x3=self.dense_block2(x2)*self.beta+x2
        x4=self.dense_block3(x3)*self.beta+x3
        x=x4*self.beta+x1
    
        x=self.leaky_1(self.bn_1(self.conv_1(x)))
        x=self.leaky_2(self.bn_2(self.conv_2(x)))
        
        x=self.leaky_3(self.bn_3(self.conv_3(x)))
        x=self.leaky_4(self.bn_4(self.conv_4(x)))
        
        x=self.leaky_5(self.bn_5(self.conv_5(x)))
        x=self.leaky_6(self.bn_6(self.conv_6(x)))
        
        x=self.leaky_7(self.bn_7(self.conv_7(x)))
        x=self.leaky_8(self.bn_8(self.conv_8(x)))
        
        x=self.leaky_9(self.bn_9(self.conv_9(x)))


  
        x=self.lst(x)
        
        return x+y





