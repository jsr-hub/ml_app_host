
import os
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for
from werkzeug import secure_filename
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import json
import numpy as np
import argparse
import imutils
import cv2
import time
import uuid
import base64
li=['axe',
'colgate',
'dabur',
'dove',
'horlicks',
'lakme',
'nestle',
'patanjali',
'revlon',
'sunsilk',
'vatica']






IMG_SHAPE = 150
model_path = 'models/m.h5'


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

def get_as_base64(url):
    return base64.b64encode(requests.get(url).content)

def predict(file2):
    
    print(file2)
    
    img_pre = cv2.imread(file2,cv2.IMREAD_UNCHANGED)
    img_pre = cv2.resize(img_pre,(IMG_SHAPE,IMG_SHAPE))
    model = tf.keras.models.load_model(model_path)
    #img_pre.astype(float)
    prediction = model.predict(np.reshape(img_pre,(-1,IMG_SHAPE,IMG_SHAPE,3)))
    print(prediction)
    print(len(prediction))
    sum=0
    maxi=-9999999;
    h=0;
    k=""
    
    for i in prediction:
        for j in i :
            if maxi<j:
                maxi=j
                k=li[h]
            h+=1
            sum+=j
    print(sum)
    return(k)

def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def template_test():
    data = {}
    data['pred'] = ''
    json_data = json.dumps(data)
    return (json_data)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        import time
        start_time = time.time()
        img = request.get_data()
        f = open('image.jpeg','w')
        f.write(img)

        if file and allowed_file(f.filename):
            filename = secure_filename(f.filename)

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(file_path)
            file.save(file_path)
            result = predict(file_path)
            print(result)
            print(file_path)
            filename = my_random_string(6) + filename

            os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("--- %s seconds ---" % str (time.time() - start_time))
            data = {}
            data['pred'] = result
            json_data = json.dumps(data)
            return (json_data)
from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

from werkzeug import SharedDataMiddleware
app.add_url_rule('/uploads/<filename>', 'uploaded_file',
                 build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads':  app.config['UPLOAD_FOLDER']
})

if __name__ == "__main__":
    app.debug=True
    app.run(host="0.0.0.0", port=3000)
