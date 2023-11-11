import os
# from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import cv2

model = load_model('ft-adam-bs32-lr1e4-do50-ep30.h5')
image_directory = 'images'

def resize(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    coba=img[0][:1]

    return coba
    
def crop(img, hoffset,woffset):
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    
    return img[woffset:width-woffset, hoffset:height-hoffset, :]

def classify_image(image_path): 

    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    # img = crop(img,48,48)
    img = np.expand_dims(img, axis=0)
    
    img /= 255.0
    after = img[0][:1]

    return img

def get_conv_layer_output(layer_name, image):
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    return intermediate_layer_model.predict(image)

# for i, layer in enumerate(model.layers):
#     print(i, layer.name)

# for layer in model.layers:
#     if 'block' in layer.name:
#         print(f"Layer: {layer.name}, Output Shape: {layer.output_shape}")

# model.summary()

# Check pixel and stride
# for layer in model.layers:
#     if 'Conv2D' in layer.__class__.__name__:  # Mengecek apakah itu lapisan konvolusi
#         print(f"Layer: {layer.name}, Pixel Size: {layer.kernel_size}, Stride: {layer.strides}")

# Get filter
for layer in model.layers:
    if 'Conv2D' in layer.__class__.__name__:  # Mengecek apakah itu lapisan konvolusi
        # Mendapatkan bobot (weights) dari lapisan konvolusi
        weights = layer.get_weights()
        
        # Bobot pertama adalah bobot filter
        filters = weights[0]
        
        # Cetak ukuran kernel dan angka-angka di dalamnya
        print(f"Layer: {layer.name}, Filter Size: {filters.shape}")
        print("Filter Values:")
        for i, filter in enumerate(filters):
            print(f"Filter {i + 1}:\n{filter}\n")


for filename in os.listdir(image_directory):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        file_path = os.path.join(image_directory, filename)
        
        img = classify_image(file_path)
        coba = resize(file_path)
        
        predictions = model.predict(img)
        
        softmax_values = predictions[0]
        class_label = np.argmax(softmax_values)
        highest_softmax = int(predictions.max() * 100)

        # conv_result = get_conv_layer_output('block1_conv1', img)
        # pooling_result = tensorflow.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_result)

        # fig, ax = plt.subplots(1, 3, figsize=(12, 4))  # Menggunakan 1 baris dan 3 kolom untuk 3 subplot

        # # Menampilkan gambar asli di subplot pertama
        # ax[0].imshow(img[0])
        # ax[0].set_title('Input Image')
        # ax[0].axis('off')

        # # Menampilkan hasil konvolusi di subplot kedua
        # ax[1].imshow(conv_result[0, :, :, 0], cmap='gray')
        # ax[1].set_title('Convolution Result')
        # ax[1].axis('off')

        # # Menampilkan hasil max pooling di subplot ketiga
        # ax[2].imshow(pooling_result[0, :, :, 0], cmap='gray')
        # ax[2].set_title('Max Pooling Result')
        # ax[2].axis('off')
        
        # # fig.savefig('uploads/matplotlib_plot.png', bbox_inches='tight', pad_inches=0, transparent=True)
        
        # plt.show()
