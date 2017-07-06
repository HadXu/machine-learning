import cv2
from keras.models import load_model
import numpy as np

video_capture = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

# 读取人脸haar模型
face_detection = cv2.CascadeClassifier('model/face_detection/haarcascade_frontalface_default.xml')

# 读取性别判断模型
gender_classifier = load_model('model/gender/simple_CNN.81-0.96.hdf5')

# 读取情绪判别模型
emotion_classifier = load_model('model/emotion/simple_CNN.530-0.65.hdf5')

gender_labels = {0: 'womam', 1: 'man'}
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                  4: 'sad', 5: 'surprise', 6: 'neutral'}

while True:
    # 读取摄像头的视频流
    _, frame = video_capture.read()

    # 将视频流转换成灰度
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸，产生坐标值
    faces = face_detection.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # 将性别判断出来
        face = frame[(y - 60):(y + h + 60), (x - 30):(x + w + 30)]

        face = cv2.resize(face, (48, 48))
        face = np.expand_dims(face, 0)
        face = face / 255.0
        gender_label_arg = np.argmax(gender_classifier.predict(face))
        gender = gender_labels[gender_label_arg]

        if gender == gender_labels[0]:
            gender_color = (255, 0, 0)
        else:
            gender_color = (0, 255, 0)

        gray_face = gray[(y - 40):(y + h + 40), (x - 20):(x + w + 20)]
        gray_face = cv2.resize(gray_face, (48, 48))
        gray_face = gray_face / 255.0
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
        emotion = emotion_labels[emotion_label_arg]

        cv2.rectangle(frame, (x, y), (x + w, y + h), gender_color, 2)
        cv2.putText(frame, gender, (x, y - 30), font, .7, gender_color, 1, cv2.LINE_AA)
        cv2.putText(frame, emotion, (x + 90, y - 30), font, .7, gender_color, 1, cv2.LINE_AA)
        cv2.imshow('face', frame)

    if cv2.waitKey(30) & ord('q') == 0xFF:
        break

# 销毁视频流
video_capture.release()
cv2.destroyAllWindows()

