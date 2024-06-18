import tensorflow as tf
from tensorflow.keras import layers, models

def psnr_metric(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def create_denoise_model(input_shape):
    inputs = layers.Input(shape=input_shape, name='input_img')
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
    conv4 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv3)
    skip1 = layers.add([conv3, conv4])
    conv5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(skip1)
    skip2 = layers.add([conv2, conv5])
    conv6 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(skip2)
    skip3 = layers.add([conv1, conv6])
    output = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(skip3)  # Change output channels to 3

    model = models.Model(inputs, output)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[psnr_metric])
    return model
