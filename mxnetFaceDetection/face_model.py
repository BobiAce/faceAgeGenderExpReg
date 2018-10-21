# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import mxnet as mx
import cv2
import sklearn.preprocessing
# from easydict import EasyDict as edict
currentPath = os.path.dirname(__file__)
sys.path.append(currentPath)
from mtcnn_detector import  MtcnnDetector
from keras.models import load_model
import face_preprocess
import time

def do_flip(data):
  for idx in xrange(data.shape[0]):
    data[idx,:,:] = np.fliplr(data[idx,:,:])

def get_model(ctx, image_size, model_str, layer):
  sym, arg_params, aux_params = mx.model.load_checkpoint(model_str, 0)
  all_layers = sym.get_internals()
  sym = all_layers[layer+'_output']
  model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
  #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
  model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
  model.set_params(arg_params, aux_params)
  return model

class FaceModel:
  def __init__(self,model,ga_model,flip,ctx = mx.cpu()):
    image_size = (112, 112)
    self.model = None
    self.ga_model = None

    if len(model)>0:
        self.model = get_model(ctx, image_size, model, 'fc1')

    if len(ga_model)>0:
        self.ga_model = get_model(ctx, image_size, ga_model, 'fc1')
    #exception model
    emotion_model_path = os.path.join(os.path.dirname(__file__), 'exceptionModel', 'fer2013_XCEPTION_0.66.hdf5')
    self.emotion_classifier = load_model(emotion_model_path, compile=False)

    self.flip = flip
    self.threshold = 1.24
    self.det_minsize = 50
    #self.det_factor = 0.9
    self.image_size = image_size
    mtcnn_path = os.path.join(os.path.dirname(__file__), 'MTCNNModel')
    detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker = 4 , accurate_landmark = False)
    self.detector = detector


  def get_input(self, face_img):
    ret = self.detector.detect_face(face_img)
    if ret is None:
      print('This image no faces..')
      return None
    bbox, points = ret
    #print(bbox)
    if bbox.shape[0]==0:
      print('This image no faces..')
      return None
    bbox = bbox[0,0:4]
    points = points[0,:].reshape((2,5)).T# matrix tranpose

    nimg = face_preprocess.preprocess(face_img, bbox, points, image_size='112,112')
    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
    aligned = np.transpose(nimg, (2,0,1))
    return aligned

  def get_feature(self, aligned):
    input_blob = np.expand_dims(aligned, axis=0)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    self.model.forward(db, is_train=False)
    embedding = self.model.get_outputs()[0].asnumpy()
    embedding = sklearn.preprocessing.normalize(embedding).flatten()
    return embedding

  def get_ga(self, aligned):
    input_blob = np.expand_dims(aligned, axis=0)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    self.ga_model.forward(db, is_train=False)
    ret = self.ga_model.get_outputs()[0].asnumpy()
    g = ret[:,0:2].flatten()
    gender = np.argmax(g)
    a = ret[:,2:202].reshape( (100,2) )
    a = np.argmax(a, axis=1)
    age = int(sum(a))
    return gender, age

  # new add
  def get_features(self, face_img):
    #face_img is bgr image
    ret = self.detector.detect_face(face_img)
    if ret is None:
      return None
    bbox, points = ret
    if bbox.shape[0]==0:
      return None
    bbox = bbox[0,0:4]
    points = points[0,:].reshape((2,5)).T
    # print(bbox)
    # print(points)
    nimg = face_preprocess.preprocess(face_img, bbox, points, image_size='112,112')
    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
    aligned = np.transpose(nimg, (2,0,1))
    #print(nimg.shape)
    embedding = None
    for flipid in [0,1]:
      if flipid==1:
          if self.flip==0:
              break
          do_flip(aligned)
      input_blob = np.expand_dims(aligned, axis=0)
      data = mx.nd.array(input_blob)
      db = mx.io.DataBatch(data=(data,))
      self.model.forward(db, is_train=False)
      _embedding = self.model.get_outputs()[0].asnumpy()
      #print(_embedding.shape)
      if embedding is None:
        embedding = _embedding
      else:
        embedding += _embedding
    embedding = sklearn.preprocessing.normalize(embedding).flatten()
    return embedding

  def cos_face(self,vector1, vector2):
    dot_product = 0.0;
    normA = 0.0;
    normB = 0.0;
    for a, b in zip(vector1, vector2):
      dot_product += a * b
      normA += a ** 2
      normB += b ** 2
    if normA == 0.0 or normB == 0.0:
      return None
    else:
      cos = dot_product / ((normA * normB) ** 0.5)
      sim = 0.5 + 0.5 * cos
      return sim

  def get_slim(self,img1,img2):
    try:
        f1 = self.get_features(img1)
        f2 = self.get_features(img2)
    except Exception as e:
        print(e)
        #return 404
    if (len(f1) > 0) and (len(f2) > 0):
      sim = self.cos_face(f1,f2)
      return sim
    else:
      return 404
  #add exception reg
  def preprocess_input(self,x, v2=True):
      x = x.astype('float32')
      x = x / 255.0
      if v2:
        x = x - 0.5
        x = x * 2.0
      return x

  def predict_emotion_exception(self,crop_img):

      emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                        4: 'sad', 5: 'surprise', 6: 'neutral'}
      emotion_target_size = self.emotion_classifier.input_shape[1:3]
      face_image_gray = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
      emo_face = cv2.resize(face_image_gray, (emotion_target_size))
      emo_face = self.preprocess_input(emo_face, True)
      emo_face = np.expand_dims(emo_face, 0)
      emo_face = np.expand_dims(emo_face, -1)
      emotion_label_arg = np.argmax(self.emotion_classifier.predict(emo_face))
      emotion_text = emotion_labels[emotion_label_arg]
      return emotion_text

  def show_age_gender(self,img):
    results = self.detector.detect_face(img)
    if results is None:
      # print('This image no faces..')
      return None
    total_bbox, total_points = results
    if total_bbox.shape[0]==0:
      # print('This image no faces..')
      return None
    draw = img.copy()
    imgH,imgW,_= img.shape
    for i,bbox in enumerate(total_bbox):
        bbox = bbox[0:4]
        # exceptiom recognotion
        y1 = int(bbox[1] - 10)
        if (y1 < 0):y1 = 0
        y2 = int(bbox[3] + 10)
        if (y2 > imgH):y2 = imgH
        x1 = int(bbox[0] - 10)
        if (x1 < 0): x1 = 0
        x2 = int(bbox[2] + 10)
        if (x2 > imgW): x2 = imgW

        crop_img = img[y1:y2, x1:x2]
        biaoqing = self.predict_emotion_exception(crop_img)

        # age and gender recognition
        points = total_points[i, :].reshape((2, 5)).T
        nimg = face_preprocess.preprocess(img, bbox, points, image_size='112,112')
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        aligned = np.transpose(nimg, (2,0,1))
        gender, age = self.get_ga(aligned)

        label = "{} {} {}".format(age,"M" if gender ==1 else "F",biaoqing)

        cv2.rectangle(draw, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        if (draw.shape[1]<400):
            cv2.putText(draw, label, ((int(bbox[0])-30), (int(bbox[1])-3)), font, 0.6, (255, 0, 0), 2)
        elif (draw.shape[1] > 400 and draw.shape[1]<600):
          cv2.putText(draw, label, ((int(bbox[0]) - 30), (int(bbox[1]) - 3)), font, 0.7, (255, 0, 0), 2)
        else:
          cv2.putText(draw, label, ((int(bbox[0]) - 30), (int(bbox[1]) - 3)), font, 0.7, (255, 0, 0), 2)
    return draw,age,gender,biaoqing


