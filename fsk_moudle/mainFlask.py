# coding: utf-8
import sys
import os
facePath = os.path.join(os.path.dirname(__file__),'../','mxnetFaceDetection/')
sys.path.append(facePath)
import face_model
import numpy as np
from flask import Flask,request, render_template
from PIL import Image
from werkzeug.utils import secure_filename
import cv2
import datetime
import random

####=========================================
#生成唯一的图片的名称字符串，防止图片显示时的重名问题
####=========================================
def Pic_str():

    nowTime = datetime.datetime.now().strftime("%Y%m%d%H%M%S") # 生成当前时间
    randomNum = random.randint(0, 100) # 生成的随机整数n，其中0<=n<=100
    if randomNum <= 10:
        randomNum = str(0) + str(randomNum)
    uniqueNum = str(nowTime) + str(randomNum)
    return uniqueNum

####===========================================
####=========================================
#    删除文件夹下所有文件
####=========================================
def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)
    return
####================================================
## 选择固定格式的图片作为输入
####================================================
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
####=================================================



####================================================
##           Flask 网页前端
####================================================
app = Flask(__name__)
#Load all model
regmodelPath = facePath + 'regModel/model'
agemodelPath = facePath + 'ageGenModel/model'
model = face_model.FaceModel(regmodelPath,
                             agemodelPath,
                                 1)#flip

####================================================
##   网页响应
####================================================
@app.route('/', methods=['GET','POST'])
def face_age_recognition():
   # Check if a valid image file was uploaded

    if request.method == 'POST':

        if('file2' in request.files) and ('file3' in request.files):
            file2 = request.files['file2']  # file2_stream : binary
            file3 = request.files['file3']  # file3_stream : binary

            if allowed_file(file2.filename) and allowed_file(file3.filename):
                return reg_faces_in_image(file2, file3)

            else:
                result = "请输入列出的图片格式 png jpg jpeg gif"
                return render_template('face_recognition.html', result1=result)


        elif('file1' in request.files):

            #return redirect(request.url)
            #return render_template('face_recognition.html')

            file1 = request.files['file1']#file1_stream : binary
            if file1.filename == '' :
                #return redirect(request.url)
                result = "请输入要被识别的 包含人脸的图片"
                return render_template('face_recognition.html', result1=result)

            if file1 and allowed_file(file1.filename):
                    return reg_age_in_image(file1)
            else:
                result = "请输入列出的图片格式 png jpg jpeg gif"
                return render_template('face_recognition.html', result1=result)

        else:
            result = "请输入两张图片进行匹配识别"
            return render_template('face_recognition.html', addres='images/face-recog.jpg',result1=result)

    # If no valid image file was uploaded, show the file upload form:
    return render_template('face_recognition.html',addres='images/face-recog.jpg')

####================================================
##         传入两张图片检测识别人脸
####================================================
def res_img(img):
    height, width, _ = img.shape
    if (height >= 600) or (width >= 600):
        # print('H: %s W: %s' % (srcHeight, srcWidth))
        large = max(height, width)
        scale = 600.0 / large
        img1 = cv2.resize(img, (int(width * scale), int(height * scale)),interpolation=cv2.INTER_AREA)
    else:
        img1 = img.copy()
    return img1

def reg_faces_in_image(file2_stream,file3_stream):

    # Load the uploaded image file file1_stream-->(img)cv
    try:
        img1 = Image.open(file2_stream)
        img2 = Image.open(file3_stream)

        img1 = cv2.cvtColor(np.asarray(img1), cv2.COLOR_RGB2BGR)
        img2 = cv2.cvtColor(np.asarray(img2), cv2.COLOR_RGB2BGR)

        img1 = res_img(img1)
        img2 = res_img(img2)
    except Exception as e:
        result_err = '传入图片错误'
        return render_template('face_recognition.html', addres='images/face-recog.jpg', result1=result_err)
    try:
        sim = model.get_slim(img1, img2)
        if (sim==404):
            result = {
                "face_found_in_image": '其中至少有一张照片没有检测到人脸，请重新上传照片',
            }
            return render_template('face_recognition.html', addres='images/face-recog.jpg',
                                   result1=result['face_found_in_image'])
    except Exception as e:
        result_feature_err = '无法获取人脸特征，请重新上传图片'
        return render_template('face_recognition.html', addres='images/face-recog.jpg',result1=result_feature_err)

    if sim >0.7:
        m = '这是同一个人'
    else:
        m = '这不是同一个人'
    # Return the result as json
    result = {
        "same_people_or_not": m,
        "Similarity": sim
    }
    return render_template('face_recognition.html', addres='images/face-recog.jpg',result1=result['same_people_or_not'],
                                                                                   result2=result['Similarity'])
##=============================================================================================================================
####================================================
##         传入一张图片 至少有一个人脸识别年龄性别
####================================================
def reg_age_in_image(file1):
    try:
        file_dir = os.path.join(os.path.dirname(__file__),'static')
        FILE_DIR_SAVEIMG = os.path.join(os.path.dirname(__file__),'static','saveimages/')

        del_file(FILE_DIR_SAVEIMG)
    except Exception as e:
        error = "error 404 内部错误"
        return render_template('face_recognition.html', result1=error,addres='images/face-recog.jpg')
    try:
        fname = secure_filename(file1.filename)
        ext = fname.rsplit('.', 1)[1]
        new_filename = 'saveimages/' + Pic_str() + '.' + ext

        img_age_stream = Image.open(file1.stream)
        img_age = cv2.cvtColor(np.asarray(img_age_stream), cv2.COLOR_RGB2BGR)
        img_age = res_img(img_age)
    except Exception as e:
        error = "读入图片错误"
        return render_template('face_recognition.html', result1=error,addres='images/face-recog.jpg')

    try:
        draw,age,gender,biaoqing= model.show_age_gender(img_age)

        if (draw.all()==None):
            result_img_err = '请上传清晰的人脸图片'
            return render_template('face_recognition.html', result1=result_img_err,addres='images/face-recog.jpg')
        cv2.imwrite(os.path.join(file_dir, new_filename), draw)

        label = "{} {} years old,Emotion maybe {}".format("Boy" if gender == 1 else "Girl",age,biaoqing)
        # print(label)

        return render_template('face_recognition.html',result1=label, addres=new_filename)#, addres=new_filename

    except Exception as e:
        m = "无法识别你的年龄信息，请重新上传新的图片"
        return render_template('face_recognition.html', result1=m,addres='images/face-recog.jpg')



if __name__ == "__main__":
    reload(sys)
    sys.setdefaultencoding('utf8')
    #app.run(host='0.0.0.0', port=8561, debug=False)
    #app.run(host='localhost',port=8000,debug=True)
    app.run(debug=False)






