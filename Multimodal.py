from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
 
from tkinter.filedialog import askopenfilename
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
import cv2

from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle


main = tkinter.Tk()
main.title("An Enhanced Multi-Modal Biometric Authentication")
main.geometry("1200x1200")

global X, Y, X_train, X_test, y_train, y_test
global multimodal
global filename


def KLDAFilters():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F) #getting Gabor features
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters

def KLDAProcess(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

klda_filter = KLDAFilters()

def getTestFingerImage(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (64,64))
    img = KLDAProcess(img,klda_filter)
    return img

def getTestFaceImage(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (64,64))
    return img

def getTestIrisImage(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (64,64))
    return img

def getTestEarImage(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (64,64))
    return img

def getTestPalmImage(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (64,64))
    return img
    
def uploadDataset():
    global filename
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    text.insert(END,str(filename)+" Dataset Loaded\n\n")
    pathlabel.config(text=str(filename)+" Dataset Loaded\n\n")


def preprocessKLDAFeatures():
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    #extract all features from different dataset images
    faceX = np.load("features/faceX.txt.npy")
    faceY = np.load("features/faceY.txt.npy")
    fingerX = np.load("features/fingerX.txt.npy")
    fingerY = np.load("features/fingerY.txt.npy")
    irisX = np.load("features/irisX.txt.npy")
    irisY = np.load("features/irisY.txt.npy")
    earX = np.load("features/earX.txt.npy")
    earY = np.load("features/earY.txt.npy")
    palmX = np.load("features/palmX.txt.npy")
    palmY = np.load("features/palmY.txt.npy")
    X = np.hstack((faceX, fingerX, irisX, earX, palmX))#concatenate all features to single array
    Y = faceY
    print(X.shape)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    text.insert(END,"Preprocessing & KLDA Features extraction process completed\n\n")
    text.insert(END,"Total images found in dataset : 600\n\n")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Dataset train and test split details\n\n")
    text.insert(END,"Total 80% images used for training : 480\n")
    text.insert(END,"Total 20% images used for training : 120\n")

def trainModel():
    text.delete('1.0', END)
    global X, Y, multimodal, X_train, X_test, y_train, y_test
    if os.path.exists("model/multimodal_model.json"):
        with open('model/multimodal_model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            multimodal = model_from_json(loaded_model_json)
        json_file.close()
        multimodal.load_weights("model/multimodal_weights.h5")
        multimodal._make_predict_function()
    else:
        multimodal = Sequential()
        multimodal.add(Convolution2D(32, 3, 3, input_shape = (X.shape[1], X.shape[2], X.shape[3]), activation = 'relu'))
        multimodal.add(MaxPooling2D(pool_size = (2, 2)))
        multimodal.add(Convolution2D(32, 3, 3, activation = 'relu'))
        multimodal.add(MaxPooling2D(pool_size = (2, 2)))
        multimodal.add(Flatten())
        multimodal.add(Dense(output_dim = 256, activation = 'relu'))
        multimodal.add(Dense(output_dim = Y.shape[1], activation = 'softmax'))
        multimodal.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist =  multimodal.fit(X_train, y_train, batch_size=16, epochs=10, shuffle=True, verbose=2, validation_data=(X_test, y_test))
        multimodal.save_weights('model/multimodal_weights.h5')            
        model_json = multimodal.to_json()
        with open("model/multimodal_model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()
        f = open('model/multimodal_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    print(multimodal.summary())
    f = open('model/multimodal_history.pckl', 'rb')
    cnn_data = pickle.load(f)
    f.close()
    print(cnn_data['acc'])
    accuracy = cnn_data['acc'][3]
    text.insert(END,"Multimodal training accuracy : "+str(accuracy))         


def graph():
    f = open('model/multimodal_history.pckl', 'rb')
    cnn_data = pickle.load(f)
    f.close()
    accuracy = cnn_data['acc']
    loss = cnn_data['loss']
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Iterations/Epoch')
    plt.ylabel('Accuracy')
    plt.plot(accuracy, 'ro-', color = 'green')
    plt.plot(loss, 'ro-', color = 'orange')
    plt.legend(['Multimodal Training Accuracy', 'Multimodal Training Loss'], loc='upper left')
    plt.title('Deep Learning Multimodal Accuracy & Loss Comparison Graph')
    plt.show()


def authentication():
    global multimodal
    filename = filedialog.askdirectory(initialdir="testImages")
    finger = getTestFingerImage(filename+'/finger.jpg')
    face = getTestFaceImage(filename+'/face.jpg')
    iris = getTestIrisImage(filename+'/iris.jpg')
    ear = getTestEarImage(filename+'/ear.jpg')
    palm = getTestEarImage(filename+'/palm.jpg')
    fin = []
    fin.append(finger)
    fa = []
    fa.append(face)
    fa = np.asarray(fa)
    fin = np.asarray(fin)
    ir = []
    ir.append(iris)
    ir = np.asarray(ir)
    ea = []
    ea.append(ear)
    ea = np.asarray(ea)
    pa = []
    pa.append(palm)
    pa = np.asarray(pa)
    test = np.hstack((fa, fin, ir, ea, pa))
    test = test.astype('float32')
    test = test/255
    print(test.shape)
    predict = multimodal.predict(test)
    predict = np.argmax(predict) + 1

    finger = cv2.imread(filename+'/finger.jpg')
    finger = cv2.resize(finger, (128,128))
    face = cv2.imread(filename+'/face.jpg')
    face = cv2.resize(face, (128,128))
    iris = cv2.imread(filename+'/iris.jpg')
    iris = cv2.resize(iris, (128,128))
    ear = cv2.imread(filename+'/ear.jpg')
    ear = cv2.resize(ear, (128,128))
    palm = cv2.imread(filename+'/palm.jpg')
    palm = cv2.resize(palm, (128,128))
    img_list = []
    img_list.append(face)
    img_list.append(finger)
    img_list.append(iris)
    img_list.append(ear)
    img_list.append(palm)
    images = cv2.hconcat(img_list)
    cv2.putText(images, "Multimodal Authenticate Person ID as : "+str(predict), (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
    cv2.imshow("Authentication Result", images)
    cv2.waitKey(0)    
    

def close():
    main.destroy()

font = ('times', 14, 'bold')
title = Label(main, text='An Enhanced Multi-Modal Biometric Authentication')
title.config(bg='DarkGoldenrod1', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=5,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Multi-Modal Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=560,y=100)

preprocessButton = Button(main, text="Preprocess & KLDA Features Extraction", command=preprocessKLDAFeatures)
preprocessButton.place(x=50,y=150)
preprocessButton.config(font=font1)

hybridMLButton = Button(main, text="Train Deep Learning Multi-Modal Model", command=trainModel)
hybridMLButton.place(x=50,y=200)
hybridMLButton.config(font=font1)

snButton = Button(main, text="Training Accuracy Graph", command=graph)
snButton.place(x=50,y=250)
snButton.config(font=font1)

snButton = Button(main, text="Authentication from Multi-Modal Images", command=authentication)
snButton.place(x=50,y=300)
snButton.config(font=font1)

graphButton = Button(main, text="Exit", command=close)
graphButton.place(x=50,y=350)
graphButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=25,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=400,y=150)
text.config(font=font1)


main.config(bg='LightSteelBlue1')
main.mainloop()
