import codecs
from fileinput import filename
from posixpath import splitext
import sys
from unittest import result
import webbrowser
from flask import Flask, request, render_template ,url_for,redirect, flash, make_response
from flask_ngrok import run_with_ngrok
import json
from numpy import true_divide
import pandas as pd
import os
from werkzeug.utils import secure_filename
import subprocess
import uuid
import urllib.request
from black import main
from numpy.random import seed
seed(101)
import tensorflow as tf
tf.random.set_seed(101)


import pandas as pd
import numpy as np

import json
import argparse

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import binary_accuracy

import os
import cv2

import imageio
import skimage
import skimage.io
import skimage.transform

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools
import shutil
import matplotlib.pyplot as plt 
#import efficientnet.tfkeras
#from tensorflow.keras.models import load_model
#from tensorflow.keras.preprocessing import image as TFimage
#from tensorflow.keras import models

#from focal_loss import BinaryFocalLoss

import pickle
import numpy as np

#from tf_explain.core.grad_cam import GradCAM

#from azure.cognitiveservices.vision.customvision.prediction import CustomvisionPredictionClient

app = Flask(__name__)
run_with_ngrok(app)
WEB_APP=os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = 'static/upload/'
WAB_APP=os.path.dirname(os.path.abspath(__file__))


app = Flask(__name__)
app.secret_key = "cairocoders-ednalan"
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/index')
def index():
   return render_template('index.html')
	
# @app.route('/uploader', methods = ['GET', 'POST'])
# def upload_files():
#    if request.method == 'POST':
#       f = request.files['file']
#       filename = secure_filename(f.filename)
#       f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#       return 'file uploaded successfully'


# @app.route('/uploads/<name>')
# def download_file(name):
#     return send_from_directory(app.config["UPLOAD_FOLDER"], name)



@app.route("/show",methods = ['POST','GET'])
def show():
    data = pd.read_csv('db.csv')
    data = data.to_numpy()

    return render_template("show.html",datas= data)

@app.route('/login')
def login():
   return render_template('login.html')

@app.route('/cooklogin')
def cooklogin():
   if request.method == "POST":
      first_name = request.form.get("fname")
       # getting input with name = lname in HTML form 
      last_name = request.form.get("lname") 

      resp=make_response(render_template("login.html", name = f"{first_name} {last_name}"))
      resp.set_cookie('fname', first_name)
      resp.set_cookie('lname', last_name)
      return resp
    
   if request.method == "GET":
        getval = request.args
        print(getval)
        print(getval.get('fname'))
        print(getval.get('lname'))
      
   return render_template('login.html')

@app.route('/Preupload')
def pre11():
   return render_template('select.html')

@app.route('/Preupload', methods=["GET","POST"])
def Preupload():

   file = request.files['file']
   if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
   target=os.path.join(WEB_APP, 'static/IMAGE_UPLOADS/')
   if not os.path.isdir(target):
      os.mkdir(target)
   upload=request.files.getlist("file")[0]
   print("File name: {}".format(upload.filename))
   filename=str(uuid.uuid1())+upload.filename
    

   ext=os.path,splitext(upload.filename)[1]
   if (ext==".jpg") or (ext==".png") or (ext==".bmp") or (ext==".JPG") or (ext==".jpeg") or (ext==".JPEG") or (ext==".PNG"):
      print("File accepted")
   
   destination = os.path.join(target, filename)
   print("File saved to:", destination)
   upload.save(destination)
  
   # dbpd=dbpd.append({'file':filename},ignore_index=True)
   # dbpd.to_csv('db.csv',index=False)

   image_predict = target + filename

#def perdict(image_predict):
   from keras.models import load_model 
   model = load_model('model8_soft_pre_cate_adam.h5')

   list_image = []#os.path.join('static/IMAGE_UPLOAD/')+str

   img = cv2.imread(image_predict)
   img = cv2.resize(img, (224, 224))
   list_image.append(img)

   predict_img = np.array(list_image, dtype="float32") / 255.0

   result1 = model.predict(predict_img)

   result = np.round(result1, 3)
  
   # def ScoreCam(model, img_array, layer_name, max_N=-1):
   #    cls = np.argmax(model.predict(img_array))
   #    act_map_array = Model(inputs=model.input, outputs=model.get_layer(layer_name).output).predict(img_array)
    
   #  # extract effective maps
   #    if max_N != -1:
   #      act_map_std_list = [np.std(act_map_array[0,:,:,k]) for k in range(act_map_array.shape[3])]
   #      unsorted_max_indices = np.argpartition(-np.array(act_map_std_list), max_N)[:max_N]
   #      max_N_indices = unsorted_max_indices[np.argsort(-np.array(act_map_std_list)[unsorted_max_indices])]
   #      act_map_array = act_map_array[:,:,:,max_N_indices]

   #    input_shape = model.layers[0].output_shape[0][1:]  # get input shape
   #  # 1. upsampled to original input size
   #    act_map_resized_list = [cv2.resize(act_map_array[0,:,:,k], input_shape[:2], interpolation=cv2.INTER_LINEAR) for k in range(act_map_array.shape[3])]
   #  # 2. normalize the raw activation value in each activation map into [0, 1]
   #    act_map_normalized_list = []
   #    for act_map_resized in act_map_resized_list:
   #      if np.max(act_map_resized) - np.min(act_map_resized) != 0:
   #          act_map_normalized = act_map_resized / (np.max(act_map_resized) - np.min(act_map_resized))
   #      else:
   #          act_map_normalized = act_map_resized
   #      act_map_normalized_list.append(act_map_normalized)
   #  # 3. project highlighted area in the activation map to original input space by multiplying the normalized activation map
   #    masked_input_list = []
   #    for act_map_normalized in act_map_normalized_list:
   #      masked_input = np.copy(img_array)
   #      for k in range(3):
   #          masked_input[0,:,:,k] *= act_map_normalized
   #      masked_input_list.append(masked_input)
   #    masked_input_array = np.concatenate(masked_input_list, axis=0)
   #  # 4. feed masked inputs into CNN model and softmax
   #    pred_from_masked_input_array = softmax(model.predict(masked_input_array))
   #  # 5. define weight as the score of target class
   #    weights = pred_from_masked_input_array[:,cls]
   #  # 6. get final class discriminative localization map as linear weighted combination of all activation maps
   #    cam = np.dot(act_map_array[0,:,:,:], weights)
   #    cam = np.maximum(0, cam)  # Passing through ReLU
   #    cam /= np.max(cam)  # scale 0 to 1.0
   #    return cam

   # def superimpose(original_img_path, cam, emphasize=False):
    
   #    img_bgr = cv2.imread(original_img_path)

   #    heatmap = cv2.resize(cam, (img_bgr.shape[1], img_bgr.shape[0]))
   #    if emphasize:
   #      heatmap = sigmoid(heatmap, 50, 0.5, 1)
   #    heatmap = np.uint8(255 * heatmap)
   #    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
   #    hif = .8
   #    superimposed_img = heatmap * hif + img_bgr
   #    superimposed_img = np.minimum(superimposed_img, 255.0).astype(np.uint8)  # scale 0 to 255  
   #    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    
   #    return superimposed_img_rgb

   # def softmax(x):
   #  f = np.exp(x)/np.sum(np.exp(x), axis = 1, keepdims = True)
   #  return f

   # def sigmoid(x, a, b, c):
   #  return c / (1 + np.exp(-a * (x-b)))

   # def my_preprocess_img(image_path, size=(224,224)):
   #  train_img_Dense = []
   #  img = cv2.imread(image_path)                     
   #  x = cv2.resize(img, size)                                               # img.shape = (224, 224, 3)
   #  train_img_Dense.append(x)                                          # train_img_Dense = (1, 224, 224, 3)
   #  x = np.array(train_img_Dense, dtype="float32") / 255.0   
   #  return x 

   # img_array = my_preprocess_img(image_predict)          #read_and_preprocess_img คือฟังก์ชันที่สร้างไว้

   # relu = ScoreCam(  model  , img_array , layer_name = 'relu' ,  max_N=-1)

   # scorecam_image = superimpose(  original_img_path = image_predict  ,  cam= relu , ) 
   result1 = np.round((result[0][1])*100, 3)
   result2 = np.round((result[0][0])*100, 3)
   result3 = np.round((result[0][2])*100, 3)

   print (f' {image_predict} = Normal:  {result1}   Lung cancer: {result2}  Tuberculosis: {result3} ')
   # print(f' output form superimpose.shape {scorecam_image.shape} ')

   if request.method == "POST":
       # getting input with name = lname in HTML form 
      resp=make_response(render_template("detail.html", name = f"{filename}",result1=result1, result2=result2, result3=result3, filename=filename))
      resp.set_cookie("file", filename)
      return resp
 
   
@app.route('/display/<filename>')
def display_image(filename):
   #  return send_from_directory(app.config['UPLOAD_FOLDER'], image_predict)
   # print('display_image filename: ' + filename)
   return redirect(url_for('static', filename='IMAGE_UPLOADS/'+ filename),code=301)



   # subprocess.run(["python","modelh5.py"])
   # pro=subprocess.Popen(["python","modelh5.py","--c",str(filename)],
   # stdout=subprocess.PIPE, stderr=subprocess.PIPE)
   # stdout=pro.communicate()
   # text=str(stdout)
   # text=text.rstrip("\n")
#   return redirect(url_for('detail'))

@app.route("/select",methods = ['POST','GET'])
def select():
   return render_template('select.html')

@app.route("/feq",methods = ['POST','GET'])
def feq():
   return render_template('feq.html')

@app.route('/detail')
def detail():
   return render_template('detail.html')

@app.route('/testin')
def testin():
   return render_template('testin.html')

@app.route('/contact')
def contact1():
   return render_template('contact.html')


@app.route('/about')
def about1():
   return render_template('about.html')
# @app.route('/predict')
# def predict():
#    images=request.form.get('selected-image')
#    subprocess.run(["python","app_startup_code.js"])
#    pro=subprocess.Popen(["python","app_startup_code.js","--c",str(images)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
#    (sys.stdout,sys.stderr)=pro.communicate
#    text=str(sys.stdout,'utf-8')
#    text=text.rstrip("\n")
#    return render_template('testin.html',outs=text)

@app.route("/home1",methods = ['POST','GET'])
def home1():
#   dbpd=pd.read_csv('db.csv')
#   if request.method == "POST":
#     first_name = request.form.get("fname")
#     last_name = request.form.get("lname")
#     dbpd=dbpd.append({'fname':first_name, 'lname':last_name},ignore_index=True)
#     dbpd.to_csv('db.csv',index=False)
#     return redirect(url_for('select'))
  if request.method == "POST":
      first_name = request.form.get("fname")
       # getting input with name = lname in HTML form 
      last_name = request.form.get("lname") 
      resp=make_response(render_template("select.html", name = f"{first_name} {last_name}"))
      resp.set_cookie('fname', first_name)
      resp.set_cookie('lname', last_name)
      return resp

@app.route("/home",methods = ['POST','GET'])
def home():
   first_name=request.cookies.get('fname')
   last_name=request.cookies.get('lname')
   file_name=request.cookies.get('file')
   dbpd=pd.read_csv('db.csv')
   if request.method == "POST":
    con_firm = request.form.get("fav_language")
    in_ter = request.form.get("fav_inter")
    com_ment = request.form.get("fav_com")
    dbpd=dbpd.append({'fav_language':con_firm, 'fav_inter':in_ter, 'fav_com':com_ment, 'fname':first_name, 'lname':last_name, 'file':file_name},ignore_index=True)
    dbpd.to_csv('db.csv',index=False)
    return redirect(url_for('show'))

@app.route("/confirm",methods = ['POST','GET'])
def confirm():
    return render_template('confirm.html')


# @app.route('/setcookie', methods = ['POST', 'GET'])
# def setcookie():
#    if request.method == 'POST':
#    file = request.form.get('file')
   
#    resp = make_response(render_template('select.html'))
#    resp.set_cookie('userID',  file)
   
#    return resp
# def cookies():

#     resp = make_response(render_template("select.html"))

#     return resp
# cookies
# resp = make_response(render_template("login.html",fname=filename))
# resp.set_cookie('filename', filename)





if __name__ == "__main__":
   webbrowser.open_new('http://127.0.0.1:5000/index')
   app.run()# host ='0.0.0.0',port=5001 