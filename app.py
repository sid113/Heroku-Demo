from __future__ import division, print_function
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import cv2
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
import numpy as np
from keras.models import load_model
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import os

app = Flask(__name__)

#APP CONFIGURATIONS
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

app.config['UPLOAD_FOLDER'] = 'images'

app.config['SECRET_KEY'] = '6575fae36288be6d1bad40b2718930'

def prepare_image(path):
    """
    This function returns the numpy array of an image
    """
    img = load_img(path , target_size=(28, 28,1),grayscale=True)    
    img = img_to_array(img)    
    img = img/255
    im2arr = img.reshape(1, 28, 28, 1)

    return im2arr


@app.route('/', methods=['GET', 'POST'])

def predict():
    if request.method == 'POST':
        if request.files:
            img = request.files['file']
            extension = img.filename.split('.')[1]
            if extension not in ['jpeg', 'png', 'jpg']:
                flash('File format not suported.', 'warning')
            else:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(img.filename))
                img.save(file_path)
                preprocessed_image = prepare_image(file_path)
                global model,graph
                model = load_model('my_model_1.h5')
                graph = tf.get_default_graph()
                with graph.as_default():
                    predictions = model.predict_classes(preprocessed_image)
                labels = predictions
                flash('Predicted class for the input image is {}'.format(labels[-1]),'success')
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
