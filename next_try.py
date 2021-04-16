import cv2
import numpy as np
import pickle


labels = {}
count = 0
with open("labels.pickle","rb") as f:
    oglabels =  pickle.load(f)
tested  = {-1:0}

labels = {v: k for k,v in oglabels.items()} 
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
video = cv2.VideoCapture(0)

while True:
    
    ret,frame = video.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5)
    
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),1) #Paints eyes
        id,conf = recognizer.predict(roi_gray)
        if(conf >= 75):
            count += 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id]
            if(id not in tested):
            	tested[id]=1
            for i,val in tested.items():
            	if(i == id):
            		if(tested[id] == 10):
            			print(name,"Just entered")
            			exit()
            		else :
            			tested[id] = tested[id] + 1;
            			print(tested)
            color = (0,255,0)
            stroke = 1
            cv2.putText(frame,name,(x-10,y-10),font,1,color,stroke,cv2.LINE_AA)
        color = (255,0,0)
        stroke = 1
        width = x + w
        height = y + h
        cv2.rectangle(frame,(x,y),(width,height),color,stroke) #paints the detected frame
     
    cv2.imshow("Captured image",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    #if count==50 :
        #break

video.release()
cv2.destroyAllWindows()
