import cv2
import os

input_folder = "image_manual"
output_folder = "dataset"

faceDeteksi = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"): # Ganti ekstensi jika diperlukan
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Deteksi wajah
        faces = faceDeteksi.detectMultiScale(gray, 1.3, 5)

        for idx, (x, y, w, h) in enumerate(faces):
            # Potong gambar sesuai kotak hijau
            face_image = image[y:y+h, x:x+w]

            # Ubah gambar menjadi skala abu-abu
            face_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

            # Simpan gambar hasil dengan kotak hijau ke folder output
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, face_gray)