import cv2
import schedule
import time
import numpy as np
import face_recognition
import os
from datetime import datetime

# It will collect all the images from a folder directly to get the encodings
path = 'Identification pics'
images = []
identification = []
# usn = []
# names = []
myList = os.listdir(path)
print(myList)

# reads the image from a folder and gets the name
for name in myList:
    curImg = cv2.imread(f'{path}/{name}')
    images.append(curImg)
    identification.append(os.path.splitext(name)[0])  # It will extract only the name not the extension
print(identification)


# I have done this project for my college so i need to take the roll_no of the students
# for taking only usn
# for i in identification:
#     x = i.split(' ')
#     usn.append(x[0])
#     names.append(x[1])
# print(usn)
# print(names)


def findEncodings(images):
    encodedList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # It will convert the image to rgb form
        encodeFace = face_recognition.face_encodings(img)[0]  # It will find the encodings from the given image
        encodedList.append(encodeFace)
    return encodedList


def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        dataList = f.readlines()
        # print(dataList)
        # usnList = []
        nameList = []
        for line in dataList:
            entry = line.split(',')  # It will take comma separated entry
            # print(entry)
            # usnList.append(entry[0])
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dateString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dateString}')


# markAttendance('1SB19CS001')


encodeListKnown = findEncodings(images)
print('Encoding Complete')

capture = cv2.VideoCapture(0)  # It will capture the video from the webcam.....[0] means that only 1 cam is connected

while True:
    success, img = capture.read()
    imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # Decreasing the image size so that it may load faster
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)  # It will convert the image to rgb form

    facesCurFrame = face_recognition.face_locations(imgSmall)  # It will detect all the faces in the current frame
    encodeCurFrame = face_recognition.face_encodings(imgSmall, facesCurFrame)  # Extract encodings of the current frame

    # It this part the matching of current frame with the given data starts
    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)  # lowest value will be the best match
        # print(faceDis)
        matchIndex = np.argmin(faceDis)  # It gives the index of the person matched

        if matches[matchIndex]:
            name = identification[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = faceLoc  # To create a frame
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # to increase image frame size which we decreased earlier
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)  # Displays name
            markAttendance(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)

