import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def psnr_metric(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def denoise_img(model, img, index, flag=1, psnr_list=None):
    if psnr_list is None:
        psnr_list = []

    if index == 0:
        return img, psnr_list

    h, w, c = img.shape
    if flag == 1:
        test = model.predict(img.reshape(1, h, w, 3))
        temp = img / 255.0
        enhanced_img = temp + ((test[0] * temp) * (1 - temp))
        psnr_value = psnr_metric(tf.convert_to_tensor(img, dtype=tf.float32), tf.convert_to_tensor(enhanced_img * 255, dtype=tf.float32)).numpy()
        psnr_list.append(psnr_value)
        index -= 1
        flag = 0
        return denoise_img(model, enhanced_img, index, flag, psnr_list)
    else:
        temp = model.predict(img.reshape(1, h, w, 3))
        enhanced_img = img + ((temp[0] * img) * (1 - img))
        psnr_value = psnr_metric(tf.convert_to_tensor(img, dtype=tf.float32), tf.convert_to_tensor(enhanced_img, dtype=tf.float32)).numpy()
        psnr_list.append(psnr_value)
        index -= 1
        return denoise_img(model, enhanced_img, index, flag, psnr_list)

def compute_psnr(model, images, index):
    psnr_values = []
    for i, img in enumerate(images):
        _, psnr_list = denoise_img(model, img, index)
        psnr_values.append(psnr_list[-1])
        print(f"PSNR Value for Image {i+1}: {psnr_list[-1]}")
    return np.array(psnr_values)

def plot_denoised_img(test_low,index,img_index,model):
    Image = test_low[img_index]
    Index = index
    denoised_image, psnr_list = denoise_img(model,Image, Index, 1)
    psnr_array = np.array(psnr_list)
    psnr_values=[]
    psnr_values.append(psnr_array[Index-1])
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Actual Image", fontsize=15)
    plt.imshow(Image)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title("Denoised Image", fontsize=15)
    plt.imshow(denoised_image)
    plt.axis('off')
    plt.show()
    print("PSNR Value :", psnr_array[Index-1])
