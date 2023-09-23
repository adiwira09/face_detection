import os
import cv2
import numpy as np
from PIL import Image
import mysql.connector

conn = mysql.connector.connect(
    user='root',
    password='',
    host='localhost',
    database='face_recognition'
)
cursor = conn.cursor()

def addData():
    video = cv2.VideoCapture(0,cv2.CAP_DSHOW)

    # menggunakan algoritma Haar Cascade.
    faceDeteksi = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    id = input('Masukkan ID : ')
    nama = input('Masukkan Nama : ')
    jenis_kelamin = input('Masukkan Gender (Pria/Wanita) : ')
    domisili = input('Masukkan Domisili : ')
    
    query = f'INSERT INTO tbl_person (id, nama, jenis_kelamin, domisili) VALUES (%s, %s, %s, %s)'
    values = (id, nama, jenis_kelamin, domisili)
    try:
        cursor.execute(query,values)
        conn.commit()
        a = 0
        while True:
            a = a+1
            check, frame = video.read()
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            wajah = faceDeteksi.detectMultiScale(grey, 1.3, 5)
            for (x,y,w,h) in wajah:
                cv2.imwrite('dataset/'+str(id)+'.'+str(a)+'.jpg',grey[y:y+h,x:x+w])
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            # cv2.imshow('Webcam', frame)
            if a>29:
                break

    except Exception as error:
        print('Error: ', error)

    # Tutup koneksi dengan webcam
    video.release()

    # Tutup semua jendela OpenCV
    cv2.destroyAllWindows()

def trainImage():    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    def getImagesWithLabels(path):
        imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
        faceSamples=[]
        Ids=[]
        for imagePath in imagePaths:
            pilImage=Image.open(imagePath).convert('L')
            imageNp=np.array(pilImage,'uint8')
            Id=int(os.path.split(imagePath)[-1].split(".")[0])
            faces=detector.detectMultiScale(imageNp)
            for (x,y,w,h) in faces:
                faceSamples.append(imageNp[y:y+h,x:x+w])
                Ids.append(Id)
        return faceSamples, Ids
    faces, Ids = getImagesWithLabels('dataset')
    recognizer.train(faces, np.array(Ids))
    recognizer.save('training/training.xml')

def webcam():
    video = cv2.VideoCapture(0)

    # menggunakan algoritma Haar Cascade.
    faceDeteksi = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('training/training.xml')
    while True:
        # Ambil frame dari webcam
        check, frame = video.read()

        # Grayscale gambar
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        wajah = faceDeteksi.detectMultiScale(grey, 1.3, 5)

        for (x,y,w,h) in wajah:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            id,conf = recognizer.predict(grey[y:y+h,x:x+w])

            name = ''
            jenis_kelamin = ''
            domisili = ''

            recognized_person = None

            if conf < 70:
                cursor.execute(f'SELECT nama, jenis_kelamin, domisili FROM tbl_person WHERE id={id}')
                recognized_person = cursor.fetchone()
            if recognized_person:
                name = recognized_person[0]
                jenis_kelamin = recognized_person[1]
                domisili = recognized_person[2]
            else :
                name = 'Tidak Diketahui'
                
            cv2.putText(frame,name,(x,y-70), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0))
            cv2.putText(frame,jenis_kelamin,(x,y-40), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0))
            cv2.putText(frame,domisili,(x,y-10), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0))

        # Tampilkan frame dalam jendela
        cv2.waitKey(1)
        cv2.imshow('Webcam', frame)
        

        # Tombol 'Spasi' untuk exit webcam
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

    # Tutup koneksi dengan webcam
    video.release()

    # Tutup semua jendela OpenCV
    cv2.destroyAllWindows()
 
while True:
    os.system('cls')
    print(
    '''===== Real Time Face Detection =====
    1. Open Webcam
    2. Add Data
    0. Exit''')
    _input = int(input('Masukkan Pilihan: '))
    if _input == 1:
        webcam()
    elif _input == 2:
        addData()
        trainImage()
    elif _input == 0:
        print('==== Thank You ====')
        break
    else:
        print('Masukkan Pilihan Yang Valid')

cursor.close()
conn.close()