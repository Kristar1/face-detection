import cv2
trained_face_data= cv2.CascadeClassifier('./rec.xml')

# Image read function from opencv
# img = cv2.imread('test2.jpg')
img = cv2.imread('jr1.jpeg')


#  Must convert to grayscale
grayscaled_img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces_coordinates= trained_face_data.detectMultiScale(grayscaled_img)
print(faces_coordinates)

# Draw rectangle around the faces
(x,y,w,h)= faces_coordinates[0]
cv2.rectangle(img, (x,y), (x+w, y+w), (0,255,0),2)
(x,y,w,h)= faces_coordinates[1]
cv2.rectangle(img, (x,y), (x+w, y+w), (0,255,0),2)
(x,y,w,h)= faces_coordinates[2]
cv2.rectangle(img, (x,y), (x+w, y+w), (0,255,0),2)

cv2.imshow('Krish Face Detector', img)
cv2.waitKey()
