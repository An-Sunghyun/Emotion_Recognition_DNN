from threading import local
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
import numpy as np
from numpy import argmax
import os
import cv2
import matplotlib.pyplot as plt
import sys
from collections import OrderedDict
from numpy import exp

#얼굴 인식 라이브러리(파일) 삽입
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_coordinate = []

imageName= sys.argv[1];#이미지 이름
imagePath="./img/"+imageName

img = cv2.imread(imagePath)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
count = 0  # 반복 실행 시 덮어쓰도록 여기에 선언

for (x, y, w, h) in faces:  # 좌표 값과 rectangular의 width height를 받게 된다.
    # x,y값은 rectangular가 시작하는 지점의 좌표
    # x, y값은 좌표값 w,h는 얼굴 높이 너비
    # 원본 이미지에 얼굴의 위치를 표시하는 작업을 함.
    # for문을 돌리는 이유는 여러 개가 검출 될 수 있기 때문.

    face_coordinate_data = [x, y, w, h]
    face_coordinate.append(face_coordinate_data)
    #print(face_coordinate[count])
    # image 좌표값대로 잘라서 따로 저장하는 부분
    img_trim = img[y:y + h, x:x + w]  # trim한 결과를 img_trim에 담는다
    cv2.imwrite('target_{}.jpg'.format(count), img_trim)  # 각 카운터에 맞는 이름으로 저장

    count = count + 1

    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 얼굴에 사각형 프레임 씌우기
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]

#분석 준비
image_size=48
labels=['neutral', 'fear', 'happy', 'sad', 'angry', 'surprise']
analy_image=[]
img_count = 0

#데이터 전처리
for index in faces:
    image = cv2.imread("target_{}.jpg".format(img_count)) #이미지 읽기
    image = cv2.resize(image, (image_size,image_size)) #이미지 사이즈 편집
    image = img_to_array(image) #이미지 배열화
    #print(image.shape)
    analy_image.append(image)
    img_count += 1

analy_image = np.array(analy_image, dtype="float32") / 255.0
labels = np.array(labels)
mlb = LabelBinarizer()
labels = mlb.fit_transform(labels)
model = keras.models.load_model("./0530_adam_f32.h5")
#사진 분석(개수 상관없음)
image_predict = model.predict(analy_image)
# 분석한 결과 출력(Softmax로 정규화 된 값)
#print(image_predict)
print_count = 0
result = []
result_count = 0
result2 = []
for print_c in faces:
    neutral_data = image_predict[print_count][0]*100
    fear_data= image_predict[print_count][1]*100
    happy_data = image_predict[print_count][2]*100
    sad_data = image_predict[print_count][3]*100
    angry_data = image_predict[print_count][4]*100
    surprise_data = image_predict[print_count][5]*100

    result_temp = dict(face=result_count+1
    ,local_x=int(face_coordinate[result_count][0]), local_y=int(face_coordinate[result_count][1]), local_w=int(face_coordinate[result_count][2]), local_h=int(face_coordinate[result_count][3])
    ,neutral=round(neutral_data,2), fear=round(fear_data, 2), happy=round(happy_data,2)
    ,sad=round(sad_data,2),angry=round(angry_data,2), surprise=round(surprise_data,2))

    result.append(result_temp)

    result2_temp = {
        "face" : result_count+1,
        "local_x" : int(face_coordinate[result_count][0]),
        "local_y" : int(face_coordinate[result_count][1]),
        "local_w" : int(face_coordinate[result_count][2]),
        "local_h" : int(face_coordinate[result_count][3]),
        "happy" : round(happy_data,2),
        "neutral" : round(neutral_data,2),
        "sad" : round(sad_data,2),
        "angry" : round(angry_data,2),
        "surprise" : round(surprise_data,2),
        "fear" : round(fear_data,2),
    }
    result2.append(result2_temp)

    result_count += 1
    print_count += 1
print_count = 0

#print(result)
re = dict(data=result);
re2 = {
    "data":result_2,
}

data = json.dumps(re2)

print(data)

