import cv2 

from random import randrange

#Load pretrained data on frontal faces
trained_face_data=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


#Choose image to detect faces in...read the img into opencv
img=cv2.imread("Woo2.jpeg")

#make img be shown

#cv2.imshow("Face Detector", img)

#cv2.waitKey() #pauses code...then continues to end of program

#Make images grayscale using convertColor
grayscaled_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#cv2.imshow("Face Detector", grayscaled_img)

#cv2.waitKey() #pauses code...then continues to end of program

#Plug my gray image into open cv to detect all compositions of faces...returns coordinate of objects to draw rectangles
face_coordinates= trained_face_data.detectMultiScale(grayscaled_img)

#draw rectangles around faces 
#print(face_coordinates)  #coordinates of Xavier's face...upper left and bottom right

#do rectangle on COLORED img

#cv2.rectangle(img,(48,28), (48+ 130,28+130), (0,255,0), 2) #takes top left pt: x+width and y+height of rec, color, and thickness of rec lines....HARDCODE

for (x,y,w,h) in face_coordinates:  #loop around faces with random colors
    cv2.rectangle(img, (x,y),(x+w, y+h), (randrange(256),randrange(256),randrange(256)), 2)

#make the face detection a LOOP 

cv2.imshow("Face Detector", img)
cv2.waitKey()

###########
print("Code Completed") 