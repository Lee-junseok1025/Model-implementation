import tensorflow as tf

def h_sigmoid(x):
    return tf.nn.relu6(x + 3) / 6

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

def h_swish(x):
    return x * h_sigmoid(x)

def relu6(x):
    return tf.keras.layers.ReLU(6)(x)

def SE_block(x,hidden_num,r=8):
    y = tf.keras.layers.GlobalAveragePooling2D()(x)
    y = tf.keras.layers.Dense(hidden_num//r)(y)
    y = tf.keras.layers.Activation('relu')(y)
    y = tf.keras.layers.Dense(hidden_num)(y)
    y = tf.keras.layers.Activation('sigmoid')(y)
    y = tf.keras.layers.Reshape((1,1,y.shape[-1]))(y)
    return y

# Inverted Residual block
def conv1d_layer(inputs,filters,out_filters,k,block_id,strides=1,SE=False):
    y = tf.keras.layers.Conv2D(out_filters,kernel_size=1,strides=1,padding='same',name=f'Expansion_conv_{block_id}')(inputs)
    y = tf.keras.layers.BatchNormalization(name=f'Expansion_BN_{block_id}')(y)
    y = tf.keras.layers.Activation('relu6',name=f'Expansion_Act_{block_id}')(y)
        
    y = tf.keras.layers.DepthwiseConv2D(kernel_size=k,strides=strides,padding='same',name=f'Depth_conv_{block_id}')(y)
    y = tf.keras.layers.BatchNormalization(name=f'Depth_BN_{block_id}')(y)
    y = tf.keras.layers.Activation('relu6',name=f'Depth_Act_{block_id}')(y)
    
    if SE == True:
        a = SE_block(y,y.shape[-1])
        y = tf.keras.layers.Multiply()([a,y])
    
    y = tf.keras.layers.Conv2D(out_filters,kernel_size=1,strides=1,padding='same',name=f'Linear_conv_{block_id}')(y)
    y = tf.keras.layers.BatchNormalization(name=f'Linear_BN_{block_id}')(y)
        
    if inputs.shape[-1] == y.shape[-1] and strides == 1:
        y = tf.keras.layers.Add()([inputs,y])
    return y

# IRB+CBAM
def Conv1d_block(x,filters):
    x = tf.keras.layers.Conv2D(16,kernel_size=3,strides=2,padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(h_swish)(x)
    
    x = conv1d_layer(inputs=x,filters=16,out_filters=32,strides=2,k=3,block_id=1,SE=True)
    x = conv1d_layer(inputs=x,filters=72,out_filters=24,strides=2,k=3,block_id=2)
    x = conv1d_layer(inputs=x,filters=88,out_filters=24,strides=1,k=3,block_id=3)
    x = conv1d_layer(inputs=x,filters=96,out_filters=40,strides=2,k=5,block_id=4,SE=True)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.expand_dims(x,1)
    x = tf.expand_dims(x,2)
    x = tf.keras.layers.Conv2D(40,kernel_size=1,strides=1,padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu6')(x)
    
    x = tf.keras.layers.Conv2D(48,kernel_size=1,strides=1,padding='same')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    return x

def SE_IRCNN(filters,cnn_input,output_dim):
    raw_input = tf.keras.layers.Input(cnn_input)
    cnn_output = Conv1d_block(x=raw_input,filters=filters)
    
    output = tf.keras.layers.Dense(output_dim,activation='softmax')(cnn_output)
    model = tf.keras.Model(raw_input,output)
    return model


model = SE_IRCNN(filters=32,cnn_input=(80,80,1),output_dim=3)