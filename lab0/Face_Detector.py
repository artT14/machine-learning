from scipy.misc import face
import cv2

# load pre-trained date on face frontals from openCV (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('../outside-res/haarcascade_frontalface_default.xml')

#choose an image to detect faces in
img = cv2.imread('./bd.jpg')

#convert to greyscale
greyscaled_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#detect faces
face_coords = trained_face_data.detectMultiScale(greyscaled_img)

for face in face_coords:
    # get coords and dimensions
    (x,y,w,h) = face
    #draw rectangle around face
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

print(face_coords)


cv2.imshow('Face Detector', img)

cv2.waitKey()

print("Code Completed")