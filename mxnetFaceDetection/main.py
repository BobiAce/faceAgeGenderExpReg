# coding: utf-8
import cv2
import os
import face_model


#Load all model
model = face_model.FaceModel('regModel/model',
                             'ageGenModel/model',
                                 1)#flip
if __name__ == "__main__":

    PATH = 'imgtest/'
    imagelist = os.listdir(PATH)

    for img in imagelist:
        img_path = PATH + img
        srcimg = cv2.imread(img_path)

        srcHeight, srcWidth, _ = srcimg.shape
        if (srcHeight >= 800) or (srcWidth >= 800):
            # print('H: %s W: %s' % (srcHeight, srcWidth))
            large = max(srcHeight, srcWidth)
            scale = 800.0 / large
            srcimg = cv2.resize(srcimg, (int(srcWidth * scale), int(srcHeight * scale)))
        sim = model.get_slim(srcimg,srcimg)
        print("sim : %s"%sim)
        cv2.waitKey(0)
        # draw, age, gender = model.show_age_gender(srcimg)
        # cv2.imshow("draw",draw)









# ####================================================
# ##  curl response   reference: qixun
# ####================================================
# UPLOAD_FOLDER = '../images'
# CONTENT_TYPE = 'application/json'
# def get_response(response, status):
#     rp = Response(response=response,
#                   status=status,
#                   mimetype=CONTENT_TYPE)
#     rp.headers['Content-Type'] = CONTENT_TYPE
#     return rp
# ####================================================
# ##  generate now time
# ####================================================
# def Pic_str():
#
#     nowTime = datetime.datetime.now().strftime("%Y%m%d%H%M%S"); # 生成当前时间
#     randomNum = random.randint(0, 100); # 生成的随机整数n，其中0<=n<=100
#     if randomNum <= 10:
#       randomNum = str(0) + str(randomNum);
#     uniqueNum = str(nowTime) + str(randomNum);
#     return uniqueNum;
# ####================================================
# ##      选择固定格式的图片作为输入
# ####================================================
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
#
# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# ####=================================================
#
# app = Flask(__name__)
# ####================================================
# ##   网页响应
# ####================================================
# @app.route('/facepost', methods=['GET','POST'])
# def face_age_recognition():
#    # Check if a valid image file was uploaded
#
#     if request.method == 'POST':
#
#         if 'file1' in request.files:
#             try:
#                 file1 = request.files["file1"]  # file1_stream : binary
#             except Exception as e:
#                 return get_response("Failed load file1", 400)
#             if file1.filename == '':
#                 return get_response("Failed load file1", 400)
#             print(file1.filename)
#             if file1 and allowed_file(file1.filename):
#
#                return reg_age_in_image(file1)
#
#
#         else:
#             return get_response("Please input two images", 400)
#
#         # If no valid image file was uploaded, show the file upload form:
#         # return render_template('face_recognition.html',addres='images/face-recog.jpg')
#         result = 'please post'
#         return result
#
#
# def reg_age_in_image(file1):
#
#     # file_dir = './static/'
#     # del_file(FILE_DIR)
#     # FILE_DIR = './static/saveimages/'
#     # del_file(FILE_DIR)
#
#     # fname = secure_filename(file1.filename)
#     # ext = fname.rsplit('.', 1)[1]
#     #new_filename = 'saveimages/' + Pic_str() + '.' + ext
#
#     raw_image_bytes1 = file1.stream
#     img_age_stream = Image.open(raw_image_bytes1)
#     img_age = cv2.cvtColor(np.asarray(img_age_stream), cv2.COLOR_RGB2BGR)
#
#     height_age, width_age, _ = img_age.shape
#     if (height_age >800) or (width_age>800):
#         down_dx_age = min(800.0/height_age,800.0/width_age)
#         change_size = (int(down_dx_age*width_age),int(down_dx_age*height_age))
#         img_age = cv2.resize(img_age,change_size,interpolation=cv2.INTER_AREA)
#     try:
#         t0 = time.time()
#         results = detector.detect_face(img_age)
#         t1 = time.time()
#         print ('time for is: ', t1 - t0)
#         #label = "{}, {}".format("M" if gender == 1 else "F",age)
#         #print(label)
#         total_boxs = results[0]
#         if total_boxs is not None:
#             label = total_boxs[0]
#         r = get_response(str(label)+'\r\n',200)
#         return r
#     except Exception as e:
#         print(e)
#         return get_response("Please again upload image\r\n",400)
#
# if __name__ == "__main__":
#     reload(sys)
#     sys.setdefaultencoding('utf8')
#     #app.run(host='0.0.0.0', port=7170, debug=False)
#     #app.run(host='localhost',port=8000,debug=True)
#     app.run(debug=False)



# # =========emotion have six================##
# emotion_labels = ['angry', 'fear', 'happy', 'sad', 'surprise', 'neutral']
# #emotion_labels = ['Angry', 'Fear', 'Happy','Sad', 'Surprise', 'Neutral']
# exception_model_path = os.path.join(os.path.dirname(__file__), 'exceptionModel')
# model = Emotion.VGG_16(os.path.join(exception_model_path,'my_model_weights_83.h5'))
#
# def predict_emotion(crop_img): # a single cropped face
#     face_image_gray = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
#     #cv2.imshow("gray", face_image_gray)
#     img_emo = cv2.resize(face_image_gray, (48, 48)).astype(np.float32)
#     X = np.expand_dims(img_emo, axis=0)
#     X = np.expand_dims(X, axis=0)
#     results = model.predict(X)
#     result = results[0].tolist()
#     m = result.index(max(result))
#     for index, val in enumerate(emotion_labels):
#         if (m == index):
#             biaoqing = val
#     print (biaoqing)
#     return biaoqing




# ###===============EXCEPTION=========================
# ###===============FACE EXPRESSION===================
# emotion_model_path = os.path.join(os.path.dirname(__file__), 'exceptionModel','fer2013_XCEPTION_0.66.hdf5')
# # emotion_model_path = './exceptionModel/fer2013_mini_XCEPTION.102-0.66.hdf5'
# emotion_classifier = load_model(emotion_model_path, compile=False)
# emotion_target_size = emotion_classifier.input_shape[1:3]
# emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
#                 4: 'sad', 5: 'surprise', 6: 'neutral'}
# # emotion_offsets = (20, 40)
# # emotion_offsets = (0, 0)
# def preprocess_input(x, v2=True):
#     x = x.astype('float32')
#     x = x / 255.0
#     if v2:
#         x = x - 0.5
#         x = x * 2.0
#     return x
# def predict_emotion_xception(crop_img):
#     face_image_gray = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
#     emo_face = cv2.resize(face_image_gray,(emotion_target_size))
#     emo_face = preprocess_input(emo_face, True)
#     emo_face = np.expand_dims(emo_face, 0)
#     emo_face = np.expand_dims(emo_face, -1)
#     emotion_label_arg = np.argmax(emotion_classifier.predict(emo_face))
#     emotion_text = emotion_labels[emotion_label_arg]
#     return emotion_text
#
#
#
# if __name__ == "__main__":
#
#     #PATH = '/home/jinbo/Downloads/liudehua/'
#     PATH = '/home/jinbo/PycharmPro/mtcnn_detect/mxnet_mtcnn_face_detection/imgtest/'
#     imagelist = os.listdir(PATH)
#
#     for img in imagelist:
#         img_path = PATH + img
#         img = cv2.imread(img_path)
#         # run detector
#         results = detector.detect_face(img)
#
#         if results is not None:
#             # if results is not None:
#             #
#             #     print(results)
#             total_boxes = results[0]
#             points = results[1]
#             chips = detector.extract_image_chips(img, points, 48, 0.37)
#             draw = img.copy()
#             for i,b in enumerate(total_boxes):
#                 try:
#                     crop_img = img[int(b[1]-10):int(b[3]+10), int(b[0]-10):int(b[2]+10)]
#                     cv2.imshow('crop',crop_img)
#                     biaoqing = predict_emotion_xception(crop_img)
#                     cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255))
#                     cv2.putText(draw,biaoqing,(int(b[0]),int(b[1])),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),1)
#                 except Exception as e:
#                     print(e)
#             # extract aligned face chips
#             # chips = detector.extract_image_chips(img, points, 112, 0.37)
#             # for i, chip in enumerate(chips):
#             #     cv2.imshow('chip_'+str(i), chip)
#             #     cv2.imwrite('chip_'+str(i)+'.jpg', chip)
#             for p in points:
#                 for i in range(5):
#                     cv2.circle(draw, (p[i], p[i + 5]), 1, (0, 0, 255), 2)
#
#             cv2.imshow("detection result", draw)
#             cv2.waitKey(0)