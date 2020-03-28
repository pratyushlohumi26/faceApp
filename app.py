import numpy as np
from flask import Flask, request, jsonify, render_template, flash, redirect
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
# from retreive_pymongo_data import database

from datetime import time
import time

UPLOAD_FOLDER = '/home/megh/Desktop/pl/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
img=[]

def load_kmodel():
    model=load_model('face_reco_10.MODEL')
    model.compile(optimizer = 'adam', loss='categorical_crossentropy')
    return model

@app.route('/')
def upload_file():
    return render_template('index.html')

@app.route('/success', methods = ['POST'])  
def success():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file'] ##GETTING FILE
        file.save(file.filename) ##SAVING FILE
        # file = file.save(UPLOAD_FOLDER + '/face.jpg')
        # cap = cv2.VideoCapture('/workspace/pl/test.mp4')
        # # Read until video is completed
        # while(cap.isOpened()):
        # # Capture frame-by-frame
        #     ret, img = cap.read()
        #     if ret == True:
        #         img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
        #         frame = cv2.imencode('.jpg', img)[1].tobytes()
        #         yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        #         time.sleep(0.1)
        #     else: 
        #         break
        if file.filename == '':
            flash('No File selected')
            return redirect(request.url)
        # img = cv2.imread(file)
        flash('FILE UPLOADED')
        return redirect(request.url)
    else:
        return render_template("success.html", name = file.filename)

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    e=emb()
    all_labels = []
    #fd=face()
    label=None
    a = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0}
    #b={0:1188, 1:1184, 2:213, 3:1142, 4:1149,  5:515, 6:44, 7:969, 8:1164, 9:1059 }
    ids={"Kritika":"1188","Anuja":"1184","Nagesh":"213","Amar":"1142", "Manish":"1149", "Pratyush":"11", "Naveen":"044", "Pankaj":"969", "Shelly":"1164","Vikas":"1059"}
    people = {0:'Naveen',1:'Vikas',2:'Pratyush',3:'Amar',4:'Manish',5:'Shelly',6:'Anuja',7:'Kritika',8:'Nagesh',9:'Anurag',10:'Pankaj'}
    #abhi=None
    # data=database()
    print('attendance till now is :')
    # data.view()

    if request.method == 'POST':
        # check if the post request has the file part
        # if 'file' not in request.files:
        #     print('No file part')
        #     return jsonify({
        #         "error":"No file"
        #     })
            # return redirect(request.url)
        # file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        # if file.filename == '':
        #     print('No selected file')
        #     return jsonify({
        #         "error":"No selected file"
        #     })
            # return redirect(request.url)
        # if file:
        #     print(file.filename)
        #     file.filename = str(datetime.utcnow())+".jpg"
        #     print(file.filename)

        #     file.save(os.path.join(os.pardir, file.filename))
            # imagePath = os.path.join(os.pardir, file.filename)
            # image = cv2.imread(imagePath)
        new_people=os.listdir('new_people')
        print(new_people)
        
        for x in new_people:
            for i in os.listdir('new_people/'+x):
                img=cv2.imread('new_people'+'/'+x+'/'+i,1)
                print('new_people'+'/'+x+'/'+i)
                img=cv2.resize(img,(160,160))
                img=img.astype('float')/255.0
                img=np.expand_dims(img,axis=0)
                # image = cv2.imread('/workspace/pl/Attendance-using-Face-master/new-people/Nagesh_213/1.jpg')
                feed=e.calculate(img)
                feed=np.expand_dims(feed,axis=0)
                model = load_kmodel()
                prediction=model.predict(feed)[0]
                result=int(np.argmax(prediction))
                if(np.max(prediction)>.7):
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
    print(all_labels)
            #time.sleep(2)
    return render_template('index.html', prediction_text='Face Recognized for: {}'.format(all_labels))

@app.route('/api/v0.1/images',methods=["GET"])
def get_images():
    raw_list = mongo.db.images.find({})
    list = []
    for img in raw_list:
        print(img)
        img['_id'] = str(img['_id'])
        img['created_at'] = str(img['created_at'])
        img['updated_at'] = str(img['updated_at'])
        list.append(img)
    print(list)
    return jsonify({
        "images":list
    })

@app.route('/api/v0.1/images',methods=["PUT"])
def edit_image():
    req_data = request.get_json()
    img_name = req_data['img_name']
    new_label = "ASON"
    image = mongo.db.images.find_one({
        "name":img_name
    })
    if(image):
        myquery = { "name": img_name }
        newvalues = { "$set": { "labels": new_label, "updated_at":datetime.utcnow() } }
        mongo.db.images.update_one(myquery, newvalues)
        return jsonify({"status":"done"})
    else:
        return jsonify({"error":"no such img exists"})

if __name__ == "__main__":
    app.run(host='0.0.0.0')