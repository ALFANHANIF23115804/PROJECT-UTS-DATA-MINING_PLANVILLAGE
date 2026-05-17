import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

def channel_attention_module(x, ratio=8):
    channel = x.shape[-1]
    shared_layer_one = layers.Dense(channel//ratio, activation='relu', kernel_initializer='he_normal')
    shared_layer_two = layers.Dense(channel, kernel_initializer='he_normal')

    avg_pool = layers.GlobalAveragePooling2D()(x)
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_two(shared_layer_one(avg_pool))

    max_pool = layers.GlobalMaxPooling2D()(x)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_two(shared_layer_one(max_pool))

    cbam_feature = layers.Add()([avg_pool, max_pool])
    cbam_feature = layers.Activation('sigmoid')(cbam_feature)
    return layers.multiply([x, cbam_feature])

def spatial_attention_module(x):
    avg_pool = layers.Lambda(lambda z: tf.reduce_mean(z, axis=-1, keepdims=True))(x)
    max_pool = layers.Lambda(lambda z: tf.reduce_max(z, axis=-1, keepdims=True))(x)
    
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
    cbam_feature = layers.Conv2D(filters=1, kernel_size=7, strides=1, padding='same', activation='sigmoid')(concat)
    return layers.multiply([x, cbam_feature])

def cbam_block(x):
    x = channel_attention_module(x)
    x = spatial_attention_module(x)
    return x

def build_novel_model(input_shape=(224, 224, 3), num_classes=38):
    inputs = layers.Input(shape=input_shape)
    x = layers.Lambda(lambda z: tf.keras.applications.mobilenet_v2.preprocess_input(z))(inputs)
    base_model = MobileNetV2(input_tensor=x, include_top=False, weights='imagenet')
    
    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False

    x = base_model.output
    
    # CBAM Attention
    x = cbam_block(x)
    
    # Global pooling 
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model