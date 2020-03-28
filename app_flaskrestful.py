import numpy as np
from flask import Flask, request, jsonify, render_template, flash, redirect, make_response
from flask_restful import Resource, reqparse, Api
from flask_cors import CORS
import cv2
import numpy as np
import os
from keras.optimizers import Adam
from keras.optimizers import SGD
#from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder 
from keras.layers import Dense,Activation
from keras.layers import LeakyReLU
from keras.models import Sequential
from keras.models import load_model
from embedding import emb
from PIL import Image  
import PIL
import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = '/workspace/pl/Attendance-using-Face-master/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

api = Api(app)
model=load_model('face_reco_10.MODEL')
class basic_render(Resource):
    # @app.route('/')
    def get(self):
        headers = {'Content-Type': 'text/html'}
        return make_response(render_template('index.html'), 200, headers)

class upload(Resource):
    # @app.route('/upload',methods = ['GET','POST'])
    def post(self):
        # if request.method == 'POST':  
        f = request.files['file']  
        # f.save(f.filename)
        # filename = secure_filename(f.filename)
        f.save(os.path.join(UPLOAD_FOLDER, f.filename))
        # img = cv2.imread(f)
        imagePath = os.path.join(UPLOAD_FOLDER, f.filename)
        # return make_response(render_template('index.html'), imagepath = imagePath)
        headers = {'Content-Type': 'text/html'}
        return make_response(render_template("index.html", imagepath = imagePath),headers)
        # return render_template("index.html", imagepath = imagePath)

class predict_path(Resource):
    # def load_kmodel():
        
        # model.compile(optimizer = 'adam', loss='categorical_crossentropy')
        # return model    
    # @app.route('/predict_path',methods=['POST'])
    def get(self):
        
        e=emb()
        all_labels = []
        max_label=[]
        #fd=face()
        label=None
        a = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0}
        #b={0:1188, 1:1184, 2:213, 3:1142, 4:1149,  5:515, 6:44, 7:969, 8:1164, 9:1059 }
        ids={"Kritika":"1188","Anuja":"1184","Nagesh":"213","Amar":"1142", "Manish":"1149", "Pratyush":"11", "Naveen":"044", "Pankaj":"969", "Shelly":"1164","Vikas":"1059"}
        people = {0:'Naveen',1:'Vikas',2:'Pratyush',3:'Amar',4:'Manish',5:'Shelly',6:'Anuja',7:'Kritika',8:'Nagesh',9:'Anurag',10:'Pankaj'}
        
        # if request.method == 'POST':
        new_people=os.listdir('new_people')
        print(new_people)
        for x in new_people:
            for i in os.listdir('new_people/'+x):
                img=cv2.imread('new_people'+'/'+x+'/'+i,1)
                print('new_people'+'/'+x+'/'+i)
                img=cv2.resize(img,(160,160))
                img=img.astype('float')/255.0
                img=np.expand_dims(img,axis=0)
                    # frame = cv2.imencode('.jpg', img)[1].tobytes()
                    # yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                # image = cv2.imread('/workspace/pl/Attendance-using-Face-master/new-people/Nagesh_213/1.jpg')
                feed=e.calculate(img)
                feed=np.expand_dims(feed,axis=0)
                # model = predict_path.load_kmodel()
                prediction=model.predict(feed)[0]
                result=int(np.argmax(prediction))
                if(np.max(prediction)>.9):
                    for i in people:
                        if(result==i):
                            label=people[i]
                            # label=ids[people[i]]
                            print(label)
                            all_labels.append(label)
                            #print(label)
                            if(a[i]==0):
                                continue
                                #print(label)
                            a[i]=1
                            abhi=i

                else:
                    label='unknown'
            res = max(set(all_labels), key = all_labels.count)
            print(all_labels, res)
            all_labels=[]
            max_label.append(res)
        print(max_label)
        headers = {'Content-Type': 'text/html'}
        return make_response(render_template("index.html", prediction_text_path='Face Recognized for: {}'.format(max_label)),201,headers)
        # return make_response(render_template("index.html", profile=user_info, repos=repos), 200, headers)
        # return {'Face Recognized : ' : max_label }

class predict_upload(Resource):
    # @app.route('/predict_img',methods=['POST'])
    def get(self):
        '''
        For rendering results on HTML GUI
        '''
        e=emb()
        all_labels = []
        max_label=[]
        #fd=face()
        label=None
        a = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0}
        #b={0:1188, 1:1184, 2:213, 3:1142, 4:1149,  5:515, 6:44, 7:969, 8:1164, 9:1059 }
        ids={"Kritika":"1188","Anuja":"1184","Nagesh":"213","Amar":"1142", "Manish":"1149", "Pratyush":"11", "Naveen":"044", "Pankaj":"969", "Shelly":"1164","Vikas":"1059"}
        people = {0:'Naveen',1:'Vikas',2:'Pratyush',3:'Amar',4:'Manish',5:'Shelly',6:'Anuja',7:'Kritika',8:'Nagesh',9:'Anurag',10:'Pankaj'}
        
        # if request.method == 'POST':
        uploads=os.listdir('uploads')
        print(uploads)
        for i in os.listdir('uploads/'):
            img=cv2.imread('uploads'+'/'+i,1)
            print('uploads'+'/'+i)
            img=cv2.resize(img,(160,160))
            img=img.astype('float')/255.0
            img=np.expand_dims(img,axis=0)
                # frame = cv2.imencode('.jpg', img)[1].tobytes()
                # yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            # image = cv2.imread('/workspace/pl/Attendance-using-Face-master/new-people/Nagesh_213/1.jpg')
            feed=e.calculate(img)
            feed=np.expand_dims(feed,axis=0)
            # model = predict_path.load_kmodel()
            prediction=model.predict(feed)[0]
            result=int(np.argmax(prediction))
            if(np.max(prediction)>.9):
                for i in people:
                    if(result==i):
                        label=people[i]
                        # label=ids[people[i]]
                        print(label)
                        all_labels.append(label)
                        #print(label)
                        if(a[i]==0):
                            continue
                            #print(label)
                        a[i]=1
                        abhi=i

            else:
                label='unknown'
        res = max(set(all_labels), key = all_labels.count)
        print(all_labels, res)
        all_labels=[]
        max_label.append(res)
        print(max_label)
        headers = {'Content-Type': 'text/html'}
        # return {'Face Recognized : ' : max_label }
        return make_response(render_template('index.html', prediction_text_upload='Face Recognized for: {}'.format(max_label)),201,headers)

class test(Resource):    
    # @app.route("/test")
    def get(self):
        return jsonify({"about":"WHat Up !!"})

class test_mul10(Resource):    
    # @app.route('/test_mul10/<int:num>', methods=['GET'])
    def get(self,num):
        return jsonify({'result' : num*10})

api.add_resource(basic_render,'/')
api.add_resource(upload, '/api/v1/upload')
api.add_resource(predict_path, '/api/v1/predict_path')
api.add_resource(predict_upload, '/api/v1/predict_upload')
api.add_resource(test, '/api/v1/test')
api.add_resource(test_mul10, '/api/v1/test_mul10/<int:num>')

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, threaded=False)
    