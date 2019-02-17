import os
import face_recognition
import numpy
import pandas as pd
import csv

#zagruzhau  adressa foto iz papki
list_of_photos = os.listdir('/home/user/face_rec/photo_base/') 

#sozdau slovar' dlya encoded lic
encoded_face_dict={} 
for photo in list_of_photos:
    source = '/home/user/face_rec/photo_base/' + photo
    load_image = face_recognition.load_image_file(source) #zagruzhau foto
    encoded_face = face_recognition.face_encodings(load_image) #encodiruu ego
    encoded_face_dict[photo.split('.')[0]] = encoded_face #dobavlyau v slovar'
df = []
base = []
for Name in encoded_face_dict.keys():
    df = []
    df.append(Name)
    encoded_face = encoded_face_dict[Name][0]
    df.append(encoded_face)
    base.append(df)
    

mycolumns = ['Name', 'Encoding_Face']
df = pd.DataFrame(columns=mycolumns)
for row in base:
    df.loc[len(df)] = row

df.to_csv('base.csv')

