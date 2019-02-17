import numpy as np
import cv2
import face_recognition
import pandas as pd
import time
import os
import sys 

cap = cv2.VideoCapture(0)

for i in range(100):
    # Capture frame-by-frame
    ret, frame = cap.read()
  
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite('/home/user/face_rec/unknown_face/cam.jpg', frame)
       

df = pd.read_csv('base.csv')

Baze = {}
for j in range(len(df.Encoding_Face)):
    mass = []
    for i in df.Encoding_Face[j].strip('[').strip(']').split():
        mass.append(float(i))
    Baze[df.Name[j]] = mass

#Gruzhu lico 
if os.path.exists('/home/user/face_rec/unknown_face/cam.jpg') == True:
    known_face_encodings = face_recognition.face_encodings(face_recognition.load_image_file('/home/user/face_rec/unknown_face/cam.jpg'))
else:
    sys.exit()
    

for  Name in Baze.keys():
    result = face_recognition.api.compare_faces([Baze[Name]],known_face_encodings[0], tolerance=0.6)
    if result[0] == True:
        print('Naiden epta: '+str(Name))
        os.remove('/home/user/face_rec/unknown_face/cam.jpg')
        break
    else:
        print('Ne naiden epta')
        break


