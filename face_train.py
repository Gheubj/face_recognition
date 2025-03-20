import cv2
import os
import numpy as np
from PIL import Image 

path = os.path.dirname(os.path.abspath(__file__))

recognizer = cv2.face.LBPHFaceRecognizer_create()

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

dataPath = path+r'/dataSet'


def getImagesAndLabels(dataPath):
    imagePaths = [os.path.join(dataPath, f) for f in os.listdir(dataPath)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')  
        img_numpy = np.array(PIL_img, 'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[0].split("-")[1])
        faces = faceCascade.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)

    return faceSamples, ids

print("Обучение модели. Это может занять несколько минут. Пожалуйста, подождите...")
faces, ids = getImagesAndLabels(dataPath)
recognizer.train(faces, np.array(ids))


recognizer.write('trainer/trainer.yml')
print(f"Модель обучена и сохранена. {len(np.unique(ids))} лиц обучено.")
