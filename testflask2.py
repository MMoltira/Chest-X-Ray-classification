import codecs
from fileinput import filename
from posixpath import splitext
import sys
from unittest import result
from flask import Flask, request ,  render_template ,url_for,redirect, flash
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

@app.route('/Preupload')
def pre11():
   return render_template('select.html')

@app.route('/Preupload', methods=["GET","POST"])
def Preupload():
   dbpd=pd.read_csv('db.csv')
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
  
   dbpd=dbpd.append({'file':filename},ignore_index=True)
   dbpd.to_csv('db.csv',index=False)

   
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
   print (f' {image_predict} = Normal:  {result[0][1]}   Lung cancer: {result[0][0]}  Tuberculosis: {result[0][2]} ' )

   return render_template('select.html',result=result, filename=filename)
   
@app.route('/display/<filename>')
def display_image(filename):
   #  return send_from_directory(app.config['UPLOAD_FOLDER'], image_predict)
   # print('display_image filename: ' + filename)
   return redirect(url_for('static', filename='IMAGE_UPLOADS/'+ filename), code=301)

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
  dbpd=pd.read_csv('db.csv')
  if request.method == "POST":
    first_name = request.form.get("fname")
    last_name = request.form.get("lname")
    dbpd=dbpd.append({'fname':first_name, 'lname':last_name},ignore_index=True)
    dbpd.to_csv('db.csv',index=False)
    return redirect(url_for('select'))

@app.route("/home",methods = ['POST','GET'])
def home():
  dbpd=pd.read_csv('db.csv')
  if request.method == "POST":
    first_name = request.form.get("fname")
    last_name = request.form.get("lname")
    con_firm = request.form.get("fav_language")
    in_ter = request.form.get("fav_inter")
    com_ment = request.form.get("fav_com")
    dbpd=dbpd.append({'fname':first_name, 'lname':last_name,'fav_language':con_firm, 'fav_inter' :in_ter,'fav_com':com_ment},ignore_index=True)
    dbpd.to_csv('db.csv',index=False)
    return redirect(url_for('show'))

@app.route("/confirm",methods = ['POST','GET'])
def confirm():
   return render_template('confirm.html')

@app.route("/output",methods = ['POST','GET'])
def output():
   file = request.form.get("file")
   pro = subprocess.Popen(["python","modelh5.py","--c",str(file)],stdout=subprocess.PIPE, stderr=subprocess.PIPE)
   (stdout,stderr) = pro.communicate()
   text = codecs.getwriter('utf8')(sys.stdout)
   text
   return render_template("confirm.html",outs = text)


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
    app.run(debug = True)# host ='0.0.0.0',port=5001 
