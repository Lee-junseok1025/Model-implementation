import tensorflow as tf

def ECA(x,filters):
    y = tf.keras.layers.GlobalAveragePooling1D()(x)
    y = tf.expand_dims(y,1)
    y = tf.keras.layers.Conv1D(filters=filters,kernel_size=5,strides=1,padding='same')(y)
    y = tf.keras.layers.Activation('sigmoid')(y)
    output = tf.keras.layers.Multiply()([x,y])
    return output

def BCA_block(x,filters):
    x = tf.keras.layers.Conv1D(filters=filters,kernel_size=3,strides=1,padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.ReLU()(x)
    x = ECA(x)
    return x


def Scale_1(inputs):
    x = tf.keras.layers.Conv1D(16,kernel_size=3,strides=1,padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    
    x = ECA(x,x.shape[-1])
    x = tf.keras.layers.Conv1D(32,kernel_size=5,strides=2,padding='same')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    
    x = ECA(x,x.shape[-1])
    x = tf.keras.layers.Conv1D(64,kernel_size=3,strides=2,padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    return x


def Scale_2(inputs):
    x = tf.keras.layers.MaxPooling1D(5)(inputs)
    
    x = tf.keras.layers.Conv1D(16,kernel_size=5,strides=2,padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    
    x = ECA(x,x.shape[-1])
    x = tf.keras.layers.Conv1D(32,kernel_size=3,strides=2,padding='same')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    
    x = ECA(x,x.shape[-1])
    x = tf.keras.layers.Conv1D(64,kernel_size=3,strides=2,padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    return x


def Scale_3(inputs):
    x = tf.keras.layers.AveragePooling1D(5)(inputs)
    
    x = tf.keras.layers.Conv1D(16,kernel_size=5,strides=2,padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    
    x = ECA(x,x.shape[-1])
    x = tf.keras.layers.Conv1D(32,kernel_size=3,strides=2,padding='same')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    
    x = ECA(x,x.shape[-1])
    x = tf.keras.layers.Conv1D(64,kernel_size=3,strides=2,padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    return x

def CA_MCNN(input_shape,output_dim):
    inputs = tf.keras.layers.Input(input_shape)
    
    scale_1 = Scale_1(inputs)
    scale_2 = Scale_2(inputs)
    scale_3 = Scale_3(inputs)
    cnn_output = tf.keras.layers.Concatenate()([scale_1,scale_2,scale_3])
    x = tf.keras.layers.Dense(256)(cnn_output)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.ReLU()(x)
    output = tf.keras.layers.Dense(output_dim,activation='softmax')(x)
    model = tf.keras.Model(inputs,output)
    return model
    
    
model = CA_MCNN(input_shape=(6400,1),output_dim=3)