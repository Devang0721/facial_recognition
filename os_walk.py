import os
import cv2
from PIL import Image
import numpy as np 
import pickle

base_dir = os.path.dirname(os.path.abspath("__file__"))
image_dir = os.path.join(base_dir,"./training_data")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
i = 0
label_ids = {}
x_train = []
y_label = []

for root,dirs,files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("JPG") or file.endswith("jpeg"):
            path = os.path.join(root,file)
            label = os.path.basename(root)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            pil_image = Image.open(path).convert('L')
            img_arr = np.array(pil_image,"uint8")
            faces = face_cascade.detectMultiScale(img_arr,scaleFactor=1.2,minNeighbors=5)
        for (x,y,w,h) in faces :
            roi = img_arr[y:y+h,x:x+w]
            x_train.append(roi)
            y_label.append((current_id-1))
                
with open("labels.pickle","wb") as f:
    pickle.dump(label_ids,f)
print(label_ids)
print(y_label)
recognizer.train(x_train,np.array(y_label))
recognizer.save("trainer.yml")