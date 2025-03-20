import cv2
import numpy as np
import os
import time


trainer_path = 'trainer/trainer.yml'
if not os.path.exists(trainer_path):
    print(f"Ошибка: Файл {trainer_path} не найден.")
    exit()

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(trainer_path)
cascadePath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

cam = cv2.VideoCapture(0)

wait_time = 10
start_time = None
recognition_successful = False

while True:
    ret, img = cam.read()
    if not ret:
        print("Не удалось получить кадр с камеры.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30)
    )

    face_detected = False

    for (x, y, w, h) in faces:
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        
        
        if confidence > 35:  # 100 - 35 = 65%
            id_text = "unknown"
            confidence_text = f"  {round(100 - confidence)}%"
            color = (0, 0, 255)  
        else:
            id_text = f"Your id: {id}"  
            confidence_text = f"  {round(100 - confidence)}%"
            color = (0, 255, 0)  
            face_detected = True
            recognition_successful = True
        
        
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        
        
        cv2.putText(img, id_text, (x+5, y-5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, confidence_text, (x+5, y+h-5), font, 1, color, 1)
    
    cv2.imshow('camera', img) 
    
    
    if face_detected:
        if start_time is None:
            start_time = time.time()
        elif time.time() - start_time > wait_time:
            break
    else:
        start_time = None
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

if recognition_successful:
    print("Распознавание прошло успешно.")
else:
    print("Распознавание не прошло.")
