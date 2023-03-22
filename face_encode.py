import cv2
import face_recognition
import pickle
import os
import numpy as np

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage


cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://pbl5-94125-default-rtdb.asia-southeast1.firebasedatabase.app/",
    'storageBucket': "pbl5-94125.appspot.com"
})

bucket = storage.bucket()

# Importing student images
folderPath = 'ImagesAttendance'
blobs = bucket.list_blobs(prefix=folderPath)
# for blob in blobs:
#     print(blob.name)
imgList = []
classNames = []
for blob in blobs:

    if (blob.name == folderPath+'/'):
        continue

    # Get the Image from the storage
    array = np.frombuffer(blob.download_as_string(), np.uint8)
    imgStudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)
    imgList.append(imgStudent)

    classNames.append(blob.name.split('/')[1].split('.')[0])
print(classNames)


def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList


print("Encoding Started ...")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithNames = [encodeListKnown, classNames]
print("Encoding Complete")

file = open("EncodeFile_PBL.p", 'wb')
pickle.dump(encodeListKnownWithNames, file)
file.close()
print("File Saved")
