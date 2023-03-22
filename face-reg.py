import cv2
import numpy as np
import pickle
import face_recognition
import os
from datetime import datetime
# from PIL import ImageGrab

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://pbl5-94125-default-rtdb.asia-southeast1.firebasedatabase.app/",
    'storageBucket': "pbl5-94125.appspot.com"
})


def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Load the encoding file
print("Loading Encode File ...")
file = open('EncodeFile_PBL.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, classNames = encodeListKnownWithIds
print("Encode File Loaded")

while True:
    success, img = cap.read()

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    if facesCurFrame:
        # matches = []
        # faceDis = []
        # matchIndex = 0.0
        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(
                encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(
                encodeListKnown, encodeFace)
            # print(faceDis)
            matchIndex = np.argmin(faceDis)

            # if matches[matchIndex]:
            #     name = classNames[matchIndex].upper()
            #     # print(name)
            #     y1, x2, y2, x1 = faceLoc
            #     y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #     cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            #     cv2.putText(img, name, (x1+6, y2-6),
            #                 cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            #     markAttendance(name)

            if faceDis[matchIndex] < 0.5:
                name = classNames[matchIndex].upper()
                markAttendance(name)
                dis = round(faceDis[matchIndex], 2)
            else:
                name = 'Unknown'
                dis = -1
            # print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name + " " + str(dis), (x1+6, y2-6),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Face Attendance', img)

    cv2.waitKey(16)
