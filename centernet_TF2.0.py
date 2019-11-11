# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 15:23:42 2019

@author: ACIPLE1088
"""
import tensorflow as tf
print(tf.__version__)
l2_reg = tf.keras.regularizers.l2(0.0005)


def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))


class _conv_bn_activation(tf.keras.Model):
    def __init__(self, filters = 16 ,kernel_size = 7 ,strides=1 , activation=None):
        super().__init__()
        self.model = tf.keras.models.Sequential([
             tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding='same', kernel_initializer='he_uniform', kernel_regularizer=l2_reg),
             tf.keras.layers.BatchNormalization(),
             tf.keras.layers.Activation(activation=mish)
                                                    ])
        
    def call(self,x,training=None):
        x = self.model(x,training)
        return x




class _basic_block(tf.keras.Model):
    def __init__(self, filters,**kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.conv1 = _conv_bn_activation(self.filters, 3, 1)
        self.conv2 = _conv_bn_activation(self.filters, 3, 1)
        conv3 = _conv_bn_activation( self.filters, 1, 1)

        
    def call(self,x):
        x = self.conv1(x)
        input_channels = tf.shape(x)[-1]
        shortcut = tf.cond(tf.equal(input_channels,self.filters),
            lambda : x,
            lambda :self.conv2(x))
        '''if(input_channels==self.filters):
            shortcut = conv2
        else:
            shortcut = self.conv3(x ,training=training)'''
        return x + shortcut




class _dla_generator(tf.keras.layers.Layer):
    def __init__(self,   filters, levels, stack_block_fn,**kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.levels = levels
        self.stack_block_fn = stack_block_fn( filters)
        self.conv1 = _conv_bn_activation( filters, 3, 1)
        

    def call(self,x):
        block1 = self.stack_block_fn(x)
        aggregation = self.conv1(block1 )
        return aggregation


class CenterNet(tf.keras.Model):
    def __init__(self ,l2_reg=0.001, num_classes=3, weight_decay=None, batch_size=16, score_threshold=0.60, mode='train',**kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.score_threshold = score_threshold
        self.mode = mode
        self.l2_reg = tf.keras.regularizers.l2(0.0005)
        self.conv1 = _conv_bn_activation()
        self.basic =_basic_block(16)
        
        #self._dla_generator1 = _dla_generator(16,1,_basic_block)
        #self.model = tf.keras.models.Sequential(name='final')
        #self.model.add(basic)
    def call(self,x , training =None):
        #x = self.model(x,training = training)

        x = self.conv1(x,training = training)
        """
        If We donot build the model is gives error in tensorflow 2.0
        
        """
        self.basic.build(input_shape = x.shape)
        x = self.basic(x)
        return x

model = CenterNet()

model.build(input_shape = (None,300,300,3)) 
model.summary()



