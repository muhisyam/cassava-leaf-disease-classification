import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

from flask import Flask, request, render_template, send_from_directory
from flask_uploads import UploadSet, configure_uploads, IMAGES
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Konfigurasi untuk upload gambar
photos = UploadSet("photos", IMAGES)
app.config["UPLOADED_PHOTOS_DEST"] = "uploads"
configure_uploads(app, photos)

# Muat model VGG16
model = load_model('ft-adam-bs32-lr1e4-do50-ep30.h5')

# Fungsi untuk mendapatkan array gambar
def get_array_of_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    original_array = img[0][:1]

    return original_array

def crop(img, hoffset,woffset):
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    
    return img[woffset:width-woffset, hoffset:height-hoffset, :]

# Fungsi untuk mengklasifikasikan gambar
def classify_image(image_path):
    # Pake crop 320
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    # img = crop(img,48,48)
    img = np.expand_dims(img, axis=0)

    img /= 255.0
    rescale_array = img[0][:1]

    return img, rescale_array

def get_conv_layer_output(layer_name, image):
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    
    return intermediate_layer_model.predict(image)

def preprocessing_image(image_path, img):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    ax[0].imshow(image)
    ax[0].set_title('Original Image')

    ax[1].imshow(img[0])
    ax[1].set_title('Preprocessing Image')
    
    preprocessing_image_name = 'preprocessing_plot.png'

    fig.savefig(f'uploads/{preprocessing_image_name}', bbox_inches='tight', pad_inches=0, transparent=True)

    return preprocessing_image_name

def conv_operation_image(img, conv_result, max_pool_result):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].imshow(img[0])
    ax[0].set_title('Input Layer Image')
    ax[0].axis('off')

    ax[1].imshow(conv_result[0, :, :, 0], cmap='gray')
    ax[1].set_title('Convolution Result')
    ax[1].axis('off')

    ax[2].imshow(max_pool_result[0, :, :, 0], cmap='gray')
    ax[2].set_title('Max Pooling Result')
    ax[2].axis('off')
    
    operation_image_name = 'conv_plot.png'

    fig.savefig(f'uploads/{operation_image_name}', bbox_inches='tight', pad_inches=0, transparent=True)

    return operation_image_name

# Routing untuk halaman beranda
@app.route('/', methods=['GET', 'POST'])
def index():
    class_labels = [
        'Cassava Mosaic Disease (CMD)', 
        'Cassava Bacterial Blight (CBB)', 
        'Cassava Brown Streak Virus Disease (CBSD)', 
        'Healthy', 
        'Cassava Greem Mite (CGM)'
    ] 

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file part")

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error="No selected file")

        if file:
            # Menghapus semua file yang ada
            for filename in os.listdir('uploads'):
                file_path = os.path.join('uploads', filename)
                os.remove(file_path)

            # Save Image Name
            image_name = file.filename
            
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
            
            original_array = get_array_of_image(file_path)
            img, rescale_array = classify_image(file_path)
            conv_result = get_conv_layer_output('block1_conv1', img)
            pooling_result = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_result)
            
            # Save the Preprocessing Image
            preproc_plot_image_name = preprocessing_image(file_path, img)

            # Save the Conv Operation Image 
            op_plot_image_name = conv_operation_image(img, conv_result, pooling_result)

            predictions = model.predict(img)
            class_indices = predictions.argmax()
            # Result Classification Label
            class_label = class_labels[class_indices]
            # Highest Image Softmax Resulte
            highest_softmax = int(predictions.max() * 100)
            # All Softmax
            softmax = predictions[0]

            return render_template('index.html', image_name = image_name, class_label=class_label, result=highest_softmax, original_array=original_array, rescale_array=rescale_array, preproc_plot_image_name=preproc_plot_image_name, op_plot_image_name=op_plot_image_name, class_labels_list=class_labels, softmax=softmax)

    return render_template('index.html')

# @app.route("/uploads/<filename>")
# def get_predict_image(filename):
#     return send_from_directory(app.config["UPLOADED_PHOTOS_DEST"], filename)


@app.route("/uploads/<filename>")
def get_image(filename):
    return send_from_directory(app.config["UPLOADED_PHOTOS_DEST"], filename)



if __name__ == '__main__':
    app.run()
