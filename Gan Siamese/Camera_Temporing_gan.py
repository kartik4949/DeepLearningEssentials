from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import Conv2D,Input
from keras.layers import Flatten
from keras.layers import Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.activations import relu
from keras.layers import LeakyReLU
from keras.models import Model
from keras.layers import UpSampling2D,BatchNormalization,MaxPooling2D,Permute,Add,Lambda
from keras.utils.vis_utils import plot_model
from keras.layers import dot,Activation,Reshape
import keras.backend as K
import tensorflow as tf
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from os import listdir
import cv2
import matplotlib.pyplot as plt

class GAN:

    def __init__(self,gen_input = (16,16,256) , dis_input = (128,128,3)): 

        self.gen_input = gen_input #Generator inputs

        self.dis_input = dis_input #discriminator Inputs

        self.model_discriminator = None #discriminator model
        
        self.model_generator = None #generator model


    def Generator_model(self):
        in_layer  = Input(shape=self.gen_input)
        '''
        FIRST BLOCK OF CONV AND UPSAMPLING

        '''
        x = Conv2D(filters  = 256, 
                   kernel_size = (1,1), 
                   strides=1,
                   activation='relu', 
                   padding='same',
                   name = "con2d_1")(in_layer)
        #x = relu(x,alpha=0.0, max_value=None, threshold=0.0)
        x = UpSampling2D(size=(2, 2), 
                        data_format='channels_last', 
                        interpolation='nearest')(x)
        x = BatchNormalization(axis=-1, 
            momentum=0.99, 
            epsilon=0.001)(x)

        '''
        FIRST BLOCK OF CONV AND UPSAMPLING

        '''
        x = Conv2D(filters  = 128,
         activation='relu', 
         kernel_size = (1,1), 
         strides=1, 
         padding='same',
         name = "con2d_2")(x)
        #x = relu(x,alpha=0.0, max_value=None, threshold=0.0)
        x = UpSampling2D(size=(2, 2), 
            data_format='channels_last', 
            interpolation='nearest')(x)
        x = BatchNormalization(axis=-1, 
            momentum=0.99, 
            epsilon=0.001)(x)

        '''
        FIRST BLOCK OF CONV AND UPSAMPLING

        '''
        x = Conv2D(filters  = 128, 
            kernel_size = (1,1),
             strides=1,
             activation='relu',
              padding='same',
              name = "con2d_3")(x)
        #x = relu(x,alpha=0.0, max_value=None, threshold=0.0)
        x = UpSampling2D(size=(2, 2), 
            data_format='channels_last', 
            interpolation='nearest')(x)
        x = BatchNormalization(axis=-1, 
            momentum=0.99, 
            epsilon=0.001)(x)
        x = Conv2D(filters  = 3, 
            kernel_size = (1,1),
             strides=1,
              padding='same',
              activation="relu",
              name = "con2d_4")(x)

        self.model_generator = Model(inputs=in_layer ,outputs=x ) #MODEL INITIALIZED


        return self.model_generator


    def discriminator_model(self,input_shape = (128,128 , 3)):
        '''
        INPUT LAYER


        '''
        assert input_shape[0]>100 and input_shape[1]>100 , 'Height and width should be greater than 100 px'
        in_layer = Input(shape = input_shape)
        x = Conv2D(filters  = 32, 
            kernel_size = (1,1), 
            strides=2, 
            padding='same',name = "con2d_1")(in_layer)
        x= LeakyReLU(alpha=0.2)(x)
        x= Dropout(0.25)(x)
        x = BatchNormalization(axis=-1, 
            momentum=0.99, 
            epsilon=0.001)(x)

        '''
        SECOND BLOCK


        '''


        x = Conv2D(filters  = 64, 
            kernel_size = (1,1), 
            strides=2, 
            padding='same',name = "con2d_2")(x)
        x= LeakyReLU(alpha=0.2)(x)
        x= Dropout(0.25)(x)
        x = BatchNormalization(axis=-1, 
            momentum=0.99, 
            epsilon=0.001)(x)

        '''
        THIRD BLOCK



        '''
        x = Conv2D(filters  = 128,
         kernel_size = (1,1), 
         strides=2, 
         padding='same',name = "con2d_3")(x)
        x= LeakyReLU(alpha=0.2)(x)
        x= Dropout(0.25)(x)
        x = BatchNormalization(axis=-1, 
            momentum=0.99,
             epsilon=0.001)(x)


        '''
        FOURTH BLOCK



        '''

        x = Conv2D(filters  = 256, 
            kernel_size = (1,1), strides=1, 
            padding='same',name = "con2d_4")(x)

        x= LeakyReLU(alpha=0.2)(x)

        x= Dropout(0.25)(x)

        x = Flatten()(x)

        x = Dense(100,activation='relu')(x)
        '''
        OUTPUT



        '''
        x = Dense(1,activation='sigmoid')(x)
        model_discriminator = Model(inputs = in_layer,outputs = x)#DICRIMINATOR MODEL INTILIZAED
        opt = Adam(lr=0.0002, beta_1=0.5)
        model_discriminator.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        print(model_discriminator)
        return model_discriminator

    def Gan_Model(self,g_model,d_model):
      d_model.trainable = False
      
      model = Sequential()
      
      model.add(g_model)
      model.add(d_model)
      
      model.summary()

      adam_optimizer = Adam(lr = 0.0001 )
      model.compile(loss= 'binary_crossentropy' , optimizer = adam_optimizer )
      print('Gan_model Compiled ')
      return model




'''

METHODS TO CALCULATE
COSINE SIMILARITIES 
(PAIRWISE)
'''
def l2_norm(x, axis=None):

    square_sum = K.sum(K.square(x), axis=axis, keepdims=True)

    norm = K.sqrt(K.maximum(square_sum, K.epsilon()))

    return norm

def pairwise_cosine_sim(A_B):

    A_tensor, B_tensor = A_B

    A_mag = l2_norm(A_tensor, axis=1)

    B_mag = l2_norm(B_tensor, axis=1)

    num = K.batch_dot(A_tensor, K.permute_dimensions(B_tensor, (0,2,1)))

    den = (A_mag * K.permute_dimensions(B_mag, (0,2,1)))

    dist_mat =  num / den

    return dist_mat
def my_fun(x):
    return cosine_similarity(x[0] , x[1])
def cosine_distance(vests):
    x, y = vests
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=0, keepdims=True)

def cos_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape2)



'''
SIAMESE NETWORK FOR IMAGE TEMPORING DETECTION

'''
class SiameseNet(GAN):
    def __init__(self):
        self.model = None
        super().__init__()
    def TwinModel(self,input_shape):

        in_layer = Conv2D(filters  = 100,
         kernel_size = (4,4),
          strides=1,
          activation='relu',
           padding='same',
           input_shape = input_shape ,  name = "con2d_1")
        x =  MaxPooling2D((2,2),strides = 2)(in_layer)
        x = Conv2D(filters  = 100, 
            kernel_size = (4,4), 
            strides=1,
            activation='relu',
             padding='same',name = "con2d_2")(x)
        x = MaxPooling2D((2,2),strides = 2)(x)
        x = Conv2D(filters  = 200,
         kernel_size = (1,1),
          strides=1,
          activation='relu',
           padding='same',name = "con2d_3")(x)
        x = Flatten()(x)
        x = Dense(500)(x)
        model = Model(inputs = in_layer ,outputs = x)
        return model
    def SiameseModel(self ):

        input_shape = self.dis_input
        model = Sequential()
        model.add(Conv2D(filters  = 100, 
            kernel_size = (4,4),
             strides=1,
             activation='relu',
              padding='same',
              input_shape = input_shape ,
                name = "con2d_1"))
        model.add(MaxPooling2D((2,2),strides = 2))
        model.add(Conv2D(filters  = 100,
         kernel_size = (4,4), 
         strides=1,
         activation='relu',
          padding='same',
          name = "con2d_2"))

        model.add(MaxPooling2D((2,2),strides = 2))

        model.add(Conv2D(filters  = 200, 
            kernel_size = (2,2), strides=1,
            activation='relu',
             padding='same',name = "con2d_3"))

        model.add(Flatten())
        
        model.add(Dense(500))

        cctv_input = Input(shape = (128,128,3))
        gan_input = Input(shape = (128,128,3))


        cctv_network_side = model(cctv_input)

        gan_network_side = model(gan_input)


        #x = dot([cctv_network_side , gan_network_side]  ,axes = 1, normalize = True)
        x =  Lambda(cosine_distance , output_shape = cos_dist_output_shape)([cctv_network_side , gan_network_side])
        #x = Lambda(pairwise_cosine_sim)([cctv_network_side , gan_network_side])
        #x = Permute((1,0))(x)
        x = Dense(256)(x)
        x = Dropout(0.4)(x)
        x = Activation(activation = 'relu')(x)
        x = Dense(100)(x)
        x = Dense(50)(x)
        output  = Dense(1 , activation = 'sigmoid')(x)

        self.model = Model(inputs = [cctv_input , gan_input] , outputs = output)
        return self.model



g = GAN()
#model = g.define_discriminator()
g = GAN()
Generator_model = g.Generator_model()
discriminator_model = g.discriminator_model()
s = SiameseNet()
model = s.SiameseModel()
#model.summary()
#plot_model(model, to_file='Siamese_plot.png', show_shapes=True, show_layer_names=True)
#g.model_discriminator.summary()
discriminator_model.summary()
#model.summary()








"""




TRAINING OF GAN NETWORK TO MAKE SYNTHESIZED REFERENCE IMAGES

"""


class Train_Gan():
  def __init__(self,g_model, d_model, gan_model):
    self.Train_Data_Generator = None
    self.g_model = g_model
    self.d_model = d_model
    self.gan_model = gan_model

  def __enter__(self):
    pass

  def __exit__(self,typexe):

    pass
  def Generate_Data(self,Data,n_samples):
    ix = np.random.randint(0,len(Data) , n_samples)
    X = Data[ix]
    y = np.ones((n_samples ,1 ))

    latent_random_values = np.random.rand(n_samples , 16,16,256)
    print('generator predicting')
    X_gen = self.g_model.predict(latent_random_values)
    print('done!!!!!!!!!!!!!!!!!!')
    y_gen = np.zeros((n_samples,1))

    xtrain_batch ,ytrain_batch = np.vstack((X_gen,X)),np.vstack((y_gen,y))

    return xtrain_batch ,ytrain_batch



  def train(self,dataset , n_epochs=1, n_batch=2):
      bat_per_epo = int(len(dataset) / n_batch)
      half_batch = int(n_batch / 2)
      for i in range(n_epochs):
        for j in range(bat_per_epo):
          X, y  = self.Generate_Data(dataset,half_batch)
          loss_dis , _ = self.d_model.train_on_batch(X,y)
          input_gan = np.random.rand(n_batch , 16,16,256)
          y_gan = np.ones((n_batch , 1))
          loss_gan = self.gan_model.train_on_batch(input_gan , y_gan)
          
          print('░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░Loss of Discriminator : {} ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ Loss of Gan : {} ░░░░░░░░░░░░░░░░░░░░'.format(loss_dis , loss_gan))


  def Load_dataset(self,path):
    xtrain = []
    for i in listdir(path):
      x = cv2.resize(cv2.imread(path+i) , (128,128) , interpolation = cv2.INTER_AREA)
      x = x.astype('float32')
      
      xtrain.append(x)

    xtrain =  np.asarray(xtrain)
    xtrain = xtrain / 255.0
    return xtrain
  
  def generate_image_with_gan(self):
    x= self.g_model.predict(np.random.rand(1,16,16,256))
    print(x[0])
    plt.imshow(x[0])
    plt.show()






'''
TRAINING GAN STARTS



'''

g  = GAN()

Generator_model = g.Generator_model()
discriminator_model = g.discriminator_model()
gan_model = g.Gan_Model(Generator_model,discriminator_model)


train_gan = Train_Gan(Generator_model , discriminator_model , gan_model)

xtrain = train_gan.Load_dataset('Dataset/')
train_gan.train(xtrain)
train_gan.generate_image_with_gan()






