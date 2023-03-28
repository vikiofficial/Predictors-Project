from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import cv2
import numpy as np
import pandas as pd
import pickle


app = Flask(__name__)

# load the saved model from file
with open('heartmodel.pkl', 'rb') as f:
    model2 = pickle.load(f)
with open('diabetes_model.pkl', 'rb') as f:
    diabetes_model = pickle.load(f)

brain_model = load_model('model.h5')
lung_model = load_model('lung_model.h5')

brain_labels = {0: 'Glioma Tumor', 1: 'Meningnoma Tumor', 2: 'No tumor', 3: 'Pituitary Tumor'}
lung_labels = {0: 'COVID', 1: 'Lung Opacity', 2: 'Normal', 3: 'Viral Pneumonia'}

brain_model.make_predict_function()
lung_model.make_predict_function()

def predict_label(img_path):
    img2 = cv2.imread(img_path)
    image1 = cv2.resize(img2, (140, 140)) 
    image1 = cv2.cvtColor(image1,cv2.COLOR_RGB2GRAY)
    image1 = np.array(image1)
    X_training_img = image1.reshape(1, 140, 140)
    X_training_img = X_training_img.astype('float32')
    X_training_img /= 255
    p = brain_model.predict(X_training_img)
    p1 = np.argmax(p,axis=1)
    return brain_labels[p1[0]]

def predict_label_for_lung(img_path):
    img2 = cv2.imread(img_path)
    image1 = cv2.resize(img2, (140, 140)) 
    image1 = cv2.cvtColor(image1,cv2.COLOR_RGB2GRAY)
    image1 = np.array(image1)
    X_training_img = image1.reshape(1, 140, 140)
    X_training_img = X_training_img.astype('float32')
    X_training_img /= 255
    p = lung_model.predict(X_training_img)
    p1 = np.argmax(p,axis=1)
    return lung_labels[p1[0]]
    

@app.route('/', methods = ['GET'])
def home():
    return render_template('service.html')

@app.route('/navHeart', methods = ['GET'])
def navHeart():
    return render_template('heart.html')

@app.route('/navBrain', methods = ['GET'])
def navBrain():
    return render_template('brain.html')

@app.route('/navLung', methods = ['GET'])
def navLung():
    return render_template('lung.html')

@app.route('/predictor', methods = ['GET'])
def predictor():
    return render_template('service.html')


@app.route('/bootsrapjs', methods = ['GET'])
def bootsrapjs():
    return render_template('bootrsap.js')

@app.route('/image1', methods = ['GET'])
def image1():
    return render_template('s1.png')

@app.route('/navDiabetes', methods = ['GET'])
def navDiabetes():
    return render_template('diabetes.html')

@app.route('/predictBrain', methods = ['POST'])
def predictBrain():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    classification = predict_label(image_path)
    return render_template('brain.html', prediction = classification)


@app.route('/predictLung', methods = ['POST'])
def predictLung():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    classification = predict_label_for_lung(image_path)
    return render_template('lung.html', prediction = classification)


@app.route('/predictHeart', methods =['POST'])
def predictHeart():
    # Put all form entries values in a list 
    features = [float(i) for i in request.form.values()]
    # Convert features to array
    array_features = [np.array(features)]
    # Predict features
    prediction = model2.predict(array_features)
    output = prediction
    # Check the output values and retrive the result with html tag based on the value
    if output == 1:
        return render_template('heart.html', 
                               result = 'The patient is not likely to have heart disease!')
    else:
        return render_template('heart.html', 
                               result = 'The patient is likely to have heart disease!')


@app.route('/predictDiabetes', methods =['POST'])
def predictDiabetes():
    # Put all form entries values in a list 
    features = [float(i) for i in request.form.values()]
    # Convert features to array
    array_features = [np.array(features)]
    # Predict features
    prediction = diabetes_model.predict(array_features)
    output = prediction
    # Check the output values and retrive the result with html tag based on the value
    if output == 0:
        return render_template('diabetes.html', 
                               result = 'The patient is not likely to have diabetes!')
    else:
        return render_template('diabetes.html', 
                               result = 'The patient is likely to have diabetes!')


if __name__ == '__main__':
    app.run(port=5500, debug=True)