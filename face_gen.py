import cv2
import os
path = os.path.dirname(os.path.abspath(__file__))
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
i=0
offset=50
print('Сейчас мы добавим ваши в фото в базу для дальнейшего подтверждения вашей личности. \n')
name=input('Введите ваше айди: ')
video=cv2.VideoCapture(0)

while True:
    ret, im =video.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))
    
    for(x,y,w,h) in faces:
        i=i+1
        cv2.imwrite("dataSet/face-"+name +'.'+ str(i) + ".jpg", gray[y-offset:y+h+offset,x-offset:x+w+offset])
        cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
        cv2.imshow('im',im[y-offset:y+h+offset,x-offset:x+w+offset])
        cv2.waitKey(50)
    
    if i>150:
        video.release()
        cv2.destroyAllWindows()
        break
print('Ваши фото успешно добавлены в базу')
