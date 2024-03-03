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

# Inisialisasi flask app
app = Flask(__name__)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

# Img upload config
photos = UploadSet("photos", IMAGES)
app.config["UPLOADED_PHOTOS_DEST"] = "uploads"
configure_uploads(app, photos)

model = load_model('ft-adam-bs16-lr1e4-do50-ep50.h5')

def allowed_file(filename):
    if '.' not in filename or filename.rsplit('.', 1)[1].lower() not in ALLOWED_EXTENSIONS:
        return True

def crop_img(img, hoffset,woffset):
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    
    return img[woffset:width-woffset, hoffset:height-hoffset, :]

def preproc_img(image_path):
    # crop 320, not crop 224
    img = resize = load_img(image_path, target_size=(320, 320))
    img = img_to_array(img)
    img = crop = crop_img(img,48,48)
    img = np.expand_dims(img, axis=0)

    img /= 255.0
    rescale_array = img[0][:1]

    return img, resize, crop, rescale_array

def get_original_img_array(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    original_array = img[0][:1]

    return original_array

def get_conv_layer_output(layer_name, image):
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    
    return intermediate_layer_model.predict(image)

def render_preproc_img(image_path, img, resize, crop):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))

    ax[0].imshow(image)
    ax[0].set_title('Original Image')

    ax[1].imshow(resize)
    ax[1].set_title('Resize')

    ax[2].imshow(crop)
    ax[2].set_title('Crop')

    ax[3].imshow(img[0])
    ax[3].set_title('Preprocessing Result')
    
    preprocessing_image_name = 'preprocessing_plot.png'

    fig.savefig(f'uploads/{preprocessing_image_name}', bbox_inches='tight', pad_inches=0, transparent=True)

    return preprocessing_image_name

def render_conv_operation_img(img, conv_result, max_pool_result):
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
        file = request.files['file']

        # Check if not has img
        if file.filename == '':
            return render_template('index.html', error="No selected file")
            
        # Check if img in unsupported format
        if allowed_file(file.filename):
            return render_template('index.html', error="Not supported file extension")

        if file:
            for filename in os.listdir('uploads'):
                file_path = os.path.join('uploads', filename)
                os.remove(file_path)
            
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
            
            img, resize, crop, rescale_array = preproc_img(file_path)
            
            # Predict img using model
            predictions = model.predict(img)
            # Get array class position
            class_indices = predictions.argmax()
            # Get label from position
            class_label = class_labels[class_indices]

            highest_softmax = int(predictions.max() * 100)
            softmax = predictions[0]

            preproc_plot_image_name = render_preproc_img(file_path, img, resize, crop)

            conv_results = get_conv_layer_output('block1_conv1', img)
            op_plot_image_name = render_conv_operation_img(
                img,
                conv_results, 
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_results)
            )

            return render_template('index.html', image_name = file.filename, class_label=class_label, result=highest_softmax, original_array=get_original_img_array(file_path), rescale_array=rescale_array, preproc_plot_image_name=preproc_plot_image_name, op_plot_image_name=op_plot_image_name, class_labels_list=class_labels, softmax=softmax)

    return render_template('index.html')

@app.route("/uploads/<filename>")
def get_image(filename):
    return send_from_directory(app.config["UPLOADED_PHOTOS_DEST"], filename)

if __name__ == '__main__':
    app.run()
