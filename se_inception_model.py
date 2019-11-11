from keras.layers import Conv2D,BatchNormalization,Activation,Dense,GlobalAveragePooling2D,Multiply , AveragePooling2D 
from keras import backend as K
from keras.models import Input , Model
from keras.preprocessing.image import ImageDataGenerator 





class SE_INCEPTION:
	'''

	
    INITILIZE FILTERS,KERNELS,BIASES
	'''
	def __init__(self , 
		         MODEL_CLASSES  =  10,
		         MODEL_INPUT = (224 , 244 , 3) ,
		         filter_conv1 = 128 , 
		         kernel_conv1  = (5,5),  
		         SE_RATIO = 16 , 
		         incep_3X3_filter  = 10  , 
		         incep_1X1_filter = 32 , 
		         incep_5X5_filter = 64 , 
		         kernel_intializer = initializers.glorot_uniform() , 
		         bias_initializer  = initializers.Constant(value=0.2) ):

		self.SE_RATIO  = SE_RATIO
		self.MODEL_CLASSES = MODEL_CLASSES
		self.filter_conv1 = filter_conv1
		self.kernel_conv1 = kernel_conv1
		self.MODEL_INPUT  = MODEL_INPUT
		self.incep_3X3_filter = incep_3X3_filter
		self.incep_5X5_filter = incep_5X5_filter
		self.incep_1X1_filter = incep_1X1_filter
		self.kernel_intializer  = kernel_intializer 
		self.bias_regulizer  = bias_regulizer

	def __call__(self,*args , **kwargs):
		pass

	def __repr__(self):
		print(self.model.summary())
		return self.model.summary()

	def Sequeeze_Excitation_block(self , input_block ):
	    ch = K.int_shape(input_block)[-1]
	    x = GlobalAveragePooling2D()(input_block)
	    x = Dense(ch//self.SE_RATIO, activation='relu')(x)
	    x = Dense(ch, activation='sigmoid')(x)
    	return Multiply()([input_block, x])

    def INCEPTION_MODULE_V3(self ,
                            incep_1X1_filter = 64,
                            incep_3X3_filter_reduce = 128,
                            incep_3X3_filter = 32,
                            incep_5X5_filter_reduce = 128,
                            incep_5X5_filter = 64,
                            pool_proj_filter,
    	                    input_block , 
    	                    inception_name = None ):
    	'''


    	INCEPTION 1x1 CONNECTION
    	'''
    	incep_1X1_out = Conv2D(filters  = incep_1X1_filter , 
    		                   kernel = (1 , 1) , 
    		                   activation = 'relu' , 
    		                   padding = 'same' , 
    		                   strides = 1 , 
    		                   kernel_intializer = self.kernel_intializer  , 
    		                   bias_initializer = self.bias_initializer )(input_block)

    	'''


    	INCEPTION 3X3 CONNECTION
    	'''
    	incep_3X3_out = Conv2D(filters  = incep_3X3_filter_reduce , 
    		                   activation = 'relu' , 
    		                   kernel = (1 , 1) , 
    		                   padding = 'same' , 
    		                   strides = 1 , 
    		                   kernel_intializer = self.kernel_intializer  , 
    		                   bias_initializer = self.bias_initializer)(input_block)
    	incep_3X3_out = Conv2D(filters  = incep_3X3_filter , 
    		                   activation = 'relu' ,
    		                   kernel = (1 , 3) , 
    		                   strides = 1 , 
    		                   padding = 'same' , 
    		                   kernel_intializer = self.kernel_intializer  , 
    		                   bias_initializer = self.bias_initializer)(incep_3X3_out)
    	incep_3X3_out = Conv2D(filters  = incep_3X3_filter , 
    		                   activation = 'relu',  
    		                   kernel = (3 , 1) , 
    		                   strides = 1 , 
    		                   padding = 'same' , 
    		                   kernel_intializer = self.kernel_intializer  , 
    		                   bias_initializer = self.bias_initializer)(incep_3X3_out)


    	'''



    	INCEPTION 5X5 CONNCETION
    	'''
    	incep_5X5_out = Conv2D(filters  = incep_5X5_filter_reduce , 
    		                   activation = 'relu' , kernel = (1 , 1) , 
    		                   padding = 'same' , strides = 1 , 
    		                   kernel_intializer = self.kernel_intializer  , 
    		                   bias_initializer = self.bias_initializer)(input_block)
    	incep_5X5_out = Conv2D(filters  = incep_5X5_filter , 
    		                   activation = 'relu' , 
    		                   kernel = (3 , 3) , 
    		                   padding = 'same' , 
    		                   strides = 1 , 
    		                   kernel_intializer = self.kernel_intializer  , 
    		                   bias_initializer = self.bias_initializer)(incep_5X5_out)
    	incep_5X5_out = Conv2D(filters  = incep_5X5_filter , 
    		                   activation = 'relu' ,
    		                   kernel = (3 , 3) , 
    		                   padding = 'same' , 
    		                   strides = 1 , 
    		                   kernel_intializer = self.kernel_intializer  , 
    		                   bias_initializer = self.bias_initializer)(incep_5X5_out)

    	'''



    	INCEPTION POOL CONNECTION
    	'''
    	pool_out = MaxPool2D((3, 3), 
    		                  strides=(1, 1), 
    		                  padding='same')(input_block)
    	pool_out = Conv2D(filters  = pool_proj_filter , 
    		              activation = 'relu' , kernel = (1 , 1) , 
    		              padding = 'same' , 
    		              strides = 1 , 
    		              kernel_intializer = self.kernel_intializer  , 
    		              bias_initializer = self.bias_initializer )(pool_out)



    	INCEPTION_OUT = Concatenate([incep_1X1_out , incep_3X3_out , incep_5X5_out , pool_out] , axis = 3 , name = inception_name)

    	return INCEPTION_OUT

	
	def Build_Model(self ,loss = 'categorical_crossentropy' , loss_weights = [1, 0.3, 0.3] , optimizer = 'adam'  , calllback = [EarlyStopping()]):

		in_layer = Input(shape = (self.MODEL_INPUT))

		in_layer = Conv2D(filters  = self.filter_conv1 ,
		                  activation = 'relu' ,
			              kernel = self.kernel_conv1 , 
			              padding = 'same' , 
			              strides = 2 , 
			              kernel_intializer = self.kernel_intializer  , 
			              bias_initializer = self.bias_initializer )(in_layer)

		in_layer = MaxPool2D(kernel = (3, 3), strides=(2, 2), padding='same')(in_layer)


		in_layer = Conv2D(filters  = self.filter_conv2 ,
		                  activation = 'relu' ,
			              kernel = self.kernel_conv2 , 
			              padding = 'same' , 
			              strides = 1 , 
			              kernel_intializer = self.kernel_intializer  , 
			              bias_initializer = self.bias_initializer )(in_layer)



		in_layer = MaxPool2D(kernel = (3, 3), strides=(2, 2), padding='same')(in_layer)

		



		in_layer  = BatchNormalization()(in_layer)

		

		'''
		ADD INCEPTION MODULE 3

		'''

		incep_1  = self.INCEPTION_MODULE_V3(incep_1X1_filter = 64,
				                            incep_3X3_filter_reduce = 96,
				                            incep_3X3_filter = 128,
				                            incep_5X5_filter_reduce = 16,
				                            incep_5X5_filter = 32,
				                            pool_proj_filter = 32,
				    	                    input_block = in_layer , 
				    	                    inception_name = 'INCEPTION_MODULE_3a'
		                                    )

		#x_prev = incep_1



		incep_1 = self.INCEPTION_MODULE_V3(	incep_1X1_filter = 128,
				                            incep_3X3_filter_reduce = 128,
				                            incep_3X3_filter = 192,
				                            incep_5X5_filter_reduce = 32,
				                            incep_5X5_filter = 96,
				                            pool_proj_filter = 64,
				    	                    input_block = incep_1 , 
				    	                    inception_name = 'INCEPTION_MODULE_3b'
			                              )
		#incep_1 = Add([x_prev , incep_1 ]) 

		se_block_1 = self.Sequeeze_Excitation_block(incep_1)

		max_pool_layer_1 = MaxPool2D(kernel = (3, 3), strides=(2, 2), padding='same')(se_block_1)


		incep_1  = self.INCEPTION_MODULE_V3(incep_1X1_filter = 192,
				                            incep_3X3_filter_reduce = 96,
				                            incep_3X3_filter = 260,
				                            incep_5X5_filter_reduce = 16,
				                            incep_5X5_filter = 48,
				                            pool_proj_filter = 64,
				    	                    input_block = max_pool_layer_1 , 
				    	                    inception_name = 'INCEPTION_MODULE_4a')
		'''

		AUXILARY OUTPUT
		A1

		'''
		AUX1 = AveragePooling2D(kernel  =(5 ,5 ) ,strides = 3 )(incep_1)

		AUX1 = Conv2D(filters  = 128 , 
    		                   activation = 'relu' ,
    		                   kernel = (1 , 1) , 
    		                   padding = 'same' , 
    		                   )(AUX1)
        AUX1 = Dense(1024 , activation='relu')(AUX1)

        AUX1 = Dropout(0.7)(AUX1)

        AUX_OUTPUT_1 = Dense(self.MODEL_CLASSES , activation = 'softmax' , name  = 'AUXILARY_OUTPUT_1')(AUX1)




		incep_1 = self.INCEPTION_MODULE_V3(incep_1X1_filter = 160,
				                            incep_3X3_filter_reduce = 112,
				                            incep_3X3_filter = 224,
				                            incep_5X5_filter_reduce = 24,
				                            incep_5X5_filter = 64,
				                            pool_proj_filter = 64,
				    	                    input_block = incep_1 , 
				    	                    inception_name = 'INCEPTION_MODULE_4b')

		incep_1 = self.INCEPTION_MODULE_V3(incep_1X1_filter = 128	,
				                            incep_3X3_filter_reduce = 128,
				                            incep_3X3_filter = 256,
				                            incep_5X5_filter_reduce = 24,
				                            incep_5X5_filter = 64,
				                            pool_proj_filter = 64,
				    	                    input_block = incep_1 , 
				    	                    inception_name = 'INCEPTION_MODULE_4c')


		incep_1 = self.INCEPTION_MODULE_V3(incep_1X1_filter = 112,
				                            incep_3X3_filter_reduce = 144,
				                            incep_3X3_filter = 288,
				                            incep_5X5_filter_reduce = 32,
				                            incep_5X5_filter = 64,
				                            pool_proj_filter = 64,
				    	                    input_block = incep_1 , 
				    	                    inception_name = 'INCEPTION_MODULE_4d')


		'''

		AUXILARY 
		OUTPUT 2

		'''


		AUX2 = AveragePooling2D(kernel  =(5 ,5 ) ,strides = 3 )(incep_1)

		AUX2 = Conv2D(filters  = 128 , 
    		                   activation = 'relu' ,
    		                   kernel = (1 , 1) , 
    		                   padding = 'same' , 
    		                   )(AUX2)
        AUX2 = Dense(1024 , activation='relu')(AUX2)

        AUX2 = Dropout(0.7)(AUX2)

        AUX_OUTPUT_2 = Dense(self.MODEL_CLASSES , activation = 'softmax' , name  = 'AUXILARY_OUTPUT_1')(AUX2)




		incep_1 = self.INCEPTION_MODULE_V3(incep_1X1_filter = 256,
				                            incep_3X3_filter_reduce = 160,
				                            incep_3X3_filter = 320,
				                            incep_5X5_filter_reduce = 32,
				                            incep_5X5_filter = 128,
				                            pool_proj_filter = 128,
				    	                    input_block = incep_1 , 
				    	                    inception_name = 'INCEPTION_MODULE_4e')

		se_block_1 = self.Sequeeze_Excitation_block(incep_1)

		incep_1 = MaxPool2D(kernel = (3,3)  , stride = 2 , padding = 'same')(se_block_1)

		'''
		INCEPTION MODULE 5

		'''


		incep_1 = self.INCEPTION_MODULE_V3(incep_1X1_filter = 256,
				                            incep_3X3_filter_reduce = 160,
				                            incep_3X3_filter = 320,
				                            incep_5X5_filter_reduce = 32,
				                            incep_5X5_filter = 128,
				                            pool_proj_filter = 128,
				    	                    input_block = incep_1 , 
				    	                    inception_name = 'INCEPTION_MODULE_5a')

		incep_1 = self.INCEPTION_MODULE_V3(incep_1X1_filter = 384,
				                            incep_3X3_filter_reduce = 192,
				                            incep_3X3_filter = 384,
				                            incep_5X5_filter_reduce = 48,
				                            incep_5X5_filter = 128,
				                            pool_proj_filter = 128,
				    	                    input_block = incep_1 , 
				    	                    inception_name = 'INCEPTION_MODULE_4e')


		'''

		FINAL OUTPUT

		'''

		output = GlobalAveragePooling2D(name = 'final_output_global' )(incep_1)

		output = Dropout(0.4)(output)

		output  = Dense(self.MODEL_CLASSES , activation  = 'softmax')(output)

		model = Model(inputs = [in_layer] , outputs = [output , AUX_OUTPUT_1 , AUX_OUTPUT_2]  )

		model.compile(loss= [self.loss , self.loss , self.loss] , loss_weights = self.loss_weights , optimizer = self.optimizer  )

		self.model = model

		return self.model


	def get_model_summary(self):
		return self.model.summary()































