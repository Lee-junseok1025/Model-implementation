import tensorflow as tf

def h_sigmoid(x):
    return tf.nn.relu6(x + 3) / 6

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

def h_swish(x):
    return x * h_sigmoid(x)

def relu6(x):
    return tf.keras.layers.ReLU(6)(x)

def CA(inputs,filters,r=8):
    h_avg = tf.reduce_mean(inputs,1)
    w_avg = tf.reduce_mean(inputs,2)
    concat = tf.keras.layers.Concatenate()([h_avg,w_avg])
    concat = tf.expand_dims(concat,2)
    x = tf.keras.layers.Conv2D(filters=filters//r,kernel_size=1,strides=1,padding='same')(concat)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(h_swish)(x)
    
    h = tf.reduce_mean(inputs,1,keepdims=True)
    w = tf.reduce_mean(inputs,2,keepdims=True)
    h = tf.keras.layers.Conv2D(filters=filters,kernel_size=1,strides=1,padding='same')(h)
    h = tf.keras.layers.Activation('sigmoid')(h)
    w = tf.keras.layers.Conv2D(filters=filters,kernel_size=1,strides=1,padding='same')(w)
    w = tf.keras.layers.Activation('sigmoid')(w)
    output = tf.keras.layers.Multiply()([h,w])
    return output

def CA_layer(inputs,filters,out_filters,k,block_id,activation,strides=1,use_CA=True):
    y = tf.keras.layers.Conv2D(filters,kernel_size=1,strides=1,padding='same',name=f'Expansion_conv_{block_id}')(inputs)
    y = tf.keras.layers.BatchNormalization(name=f'Expansion_BN_{block_id}')(y)
    y = (tf.keras.layers.Activation(h_swish,name=f'Expansion_Act_{block_id}')(y)) if activation == 'HS'  else (tf.keras.layers.Activation(relu6,name=f'Expansion_Act_{block_id}')(y))
        
    y = tf.keras.layers.DepthwiseConv2D(kernel_size=k,strides=strides,padding='same',name=f'Depth_conv_{block_id}')(y)
    y = tf.keras.layers.BatchNormalization(name=f'Depth_BN_{block_id}')(y)
    y = (tf.keras.layers.Activation(h_swish,name=f'Depth_Act_{block_id}')(y)) if activation == 'HS'  else (tf.keras.layers.Activation(relu6,name=f'Depth_Act_{block_id}')(y))
    
    if use_CA == True:
        y = CA(inputs=y,filters=y.shape[-1])
    
    y = tf.keras.layers.Conv2D(out_filters,kernel_size=1,strides=1,padding='same',name=f'Linear_conv_{block_id}')(y)
    y = tf.keras.layers.BatchNormalization(name=f'Linear_BN_{block_id}')(y)
        
    if inputs.shape[-1] == y.shape[-1] and strides == 1:
        y = tf.keras.layers.Add()([inputs,y])
    return y

# IRB+CBAM
def CA_block(x):
    x = tf.keras.layers.Conv2D(16,kernel_size=3,strides=2,padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(h_swish)(x)
    
    x = CA_layer(inputs=x,filters=16,out_filters=16,strides=2,k=3,block_id=1,activation ='RE')
    x = CA_layer(inputs=x,filters=36,out_filters=24,strides=2,k=3,block_id=2,activation ='RE',use_CA=False)
    x = CA_layer(inputs=x,filters=44,out_filters=24,strides=1,k=3,block_id=3,activation ='RE',use_CA=False)
    x = CA_layer(inputs=x,filters=48,out_filters=40,strides=2,k=5,block_id=4,activation ='HS')
    x = CA_layer(inputs=x,filters=120,out_filters=40,strides=1,k=5,block_id=5,activation ='HS')
    x = CA_layer(inputs=x,filters=120,out_filters=40,strides=1,k=5,block_id=6,activation ='HS')
    x = CA_layer(inputs=x,filters=60,out_filters=48,strides=1,k=5,block_id=7,activation ='HS')
    x = CA_layer(inputs=x,filters=72,out_filters=48,strides=1,k=5,block_id=8,activation ='HS')
    x = CA_layer(inputs=x,filters=144,out_filters=96,strides=2,k=5,block_id=9,activation ='HS')
    x = CA_layer(inputs=x,filters=288,out_filters=96,strides=1,k=5,block_id=10,activation ='HS')
    x = CA_layer(inputs=x,filters=288,out_filters=96,strides=1,k=5,block_id=11,activation ='HS')

    x = tf.keras.layers.Conv2D(288,kernel_size=1,strides=1,padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(h_swish)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.expand_dims(x,1)
    x = tf.expand_dims(x,2)
    x = tf.keras.layers.Conv2D(512,kernel_size=1,strides=1,padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(h_swish)(x)
    return x


def CA_CNN(cnn_input,output_dim):
    raw_input = tf.keras.layers.Input(cnn_input)
    cnn_output = CA_block(x=raw_input)
    output = tf.keras.layers.Conv2D(output_dim,kernel_size=1,strides=1,padding='same',activation='softmax')(cnn_output)
    
    output = tf.keras.layers.GlobalAveragePooling2D()(output)
    model = tf.keras.Model(raw_input,output)
    return model

model = CA_CNN(32,(80,80,1),10)