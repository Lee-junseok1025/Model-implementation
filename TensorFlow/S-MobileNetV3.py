import tensorflow as tf

def h_sigmoid(x):
    return tf.nn.relu6(x + 3) / 6


def h_swish(x):
    return x * h_sigmoid(x)

def SEBlock(inputs,input_channels,r=16):
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Dense(input_channels// r)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dense(input_channels)(x)
    x = tf.keras.layers.Activation(h_sigmoid)(x)
    x = tf.keras.layers.Reshape((1,1,x.shape[-1]))(x)
    output = inputs * x
    return output

def BottleNeck(inputs, in_size, exp_size, out_size, s, SE, NL, k):
    x = tf.keras.layers.Conv2D(filters=exp_size,kernel_size=1,strides=1,padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = (tf.keras.layers.Activation(h_swish)(x) if NL == 'HS' else tf.nn.relu6(x))
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=k,strides=s,padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = (tf.keras.layers.Activation(h_swish)(x) if NL == 'HS' else tf.nn.relu6(x))
    if SE:
        x = SEBlock(inputs=x,input_channels=exp_size)
    x = tf.keras.layers.Conv2D(filters=out_size,kernel_size=1,strides=1,padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Activation(tf.keras.activations.linear)(x)
    if s == 1:
        x = tf.keras.layers.add([x,inputs])
    return x

def S_MobileNetV2(input_shape,output_shape):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(16,3,strides=(2,2),padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization(name='conv1_bn')(x)
    x = tf.keras.layers.ReLU(6, name='conv1_relu')(x)
    x = BottleNeck(x, t = 1, filters = x.shape[-1], out_channels = 16, stride = 1,block_id = 1)
    x = BottleNeck(x, t = 6, filters = x.shape[-1], out_channels = 32, stride = 2,block_id = 2)
    x = BottleNeck(x, t = 6, filters = x.shape[-1], out_channels = 64, stride = 2,block_id = 3)
    x = BottleNeck(x, t = 6, filters = x.shape[-1], out_channels = 96, stride = 2,block_id = 4)
    x = BottleNeck(x, t = 6, filters = x.shape[-1], out_channels = 96, stride = 2,block_id = 5)
    x = tf.keras.layers.Conv2D(128,kernel_size=(1,1),strides=(1,1),padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6)(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=1)(x)
    x = tf.keras.layers.Conv2D(3,kernel_size=1)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(output_shape,activation='softmax')(x)
    model = tf.keras.models.Model(inputs,outputs)
    return model

model = S_MobileNetV2(input_shape=(32,32,1),output_shape=3)