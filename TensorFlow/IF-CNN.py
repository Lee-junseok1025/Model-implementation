import tensorflow as tf

def CL(x,kernel_size,filters):
    x = tf.keras.layers.Conv2D(filters,kernel_size,strides=1,padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    return x

def CNNs(input_size,output_size):
    u_inputs = tf.keras.layers.Input(input_size)
    v_inputs = tf.keras.layers.Input(input_size)
    x = CL(u_inputs,filters=5,kernel_size=9)
    x = CL(x,filters=10,kernel_size=7)
    x = CL(x,filters=15,kernel_size=5)
    x = CL(x,filters=30,kernel_size=3)
    x = tf.keras.layers.Flatten()(x)
    u_model = tf.keras.Model(u_inputs,x)
    y = CL(v_inputs,filters=5,kernel_size=9)
    y = CL(y,filters=10,kernel_size=7)
    y = CL(y,filters=15,kernel_size=5)
    y = CL(y,filters=30,kernel_size=3)
    y = tf.keras.layers.Flatten()(y)
    v_model = tf.keras.Model(v_inputs,y)
    concat = tf.keras.layers.Concatenate()([u_model.output,v_model.output])
    outputs = tf.keras.layers.Dense(output_size,activation='softmax')(concat)
    model = tf.keras.Model([u_inputs,v_inputs],outputs)
    return model

model = CNNs(input_size=(80,80,1),output_size=3)