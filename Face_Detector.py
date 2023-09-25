import cv2 

from random import randrange

#Load pretrained data on frontal faces
trained_face_data=cv2.CascadeClassifier("/Users/tapassaha/Desktop/C_prac/CP_Vid/haarcascade_frontalface_default.xml")


#Choose image to detect faces in...read the img into opencv
#img=cv2.imread("/Users/tapassaha/Desktop/C_prac/CP_Vid/Woo2.jpeg")

#do with video of me (not added in repository)...can use any video with faces
webcam=cv2.VideoCapture("/Users/tapassaha/Desktop/C_prac/CP_Vid/ME.mov")    #arg of 0 reads from webcam


#make loop to go through each frame

while True:
    #read current frame
    successful_frame_read, frame=webcam.read()    #returns True and frame currently being read

    #Make images grayscale using convertColor
    grayscaled_img=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



    #Detect faces
    face_coordinates= trained_face_data.detectMultiScale(grayscaled_img)

    #draw rec
    for(x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x,y),(x+w,y+h), (randrange(256), randrange(256), randrange(256)), 2)

    cv2.imshow("Face Detector", frame)
    key= cv2.waitKey(1) #wait 1 ms without pressing key

    #use Q or q key instead of Force Quit
    if key==113 or key==81:
        break

#Release video capture object
webcam.release()



#make img be shown

#cv2.imshow("Face Detector", img)

#cv2.waitKey() #pauses code...then continues to end of program

#Make images grayscale using convertColor
#grayscaled_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#cv2.imshow("Face Detector", grayscaled_img)

#cv2.waitKey() #pauses code...then continues to end of program

#Plug my gray image into open cv to detect all compositions of faces...returns coordinate of objects to draw rectangles
#face_coordinates= trained_face_data.detectMultiScale(grayscaled_img)

#draw rectangles around faces 
#print(face_coordinates)  #coordinates of Xavier's face...upper left and bottom right

#do rectangle on COLORED img

#cv2.rectangle(img,(48,28), (48+ 130,28+130), (0,255,0), 2) #takes top left pt: x+width and y+height of rec, color, and thickness of rec lines....HARDCODE

'''
for (x,y,w,h) in face_coordinates:  #loop around faces with random colors
    cv2.rectangle(img, (x,y),(x+w, y+h), (randrange(256),randrange(256),randrange(256)), 2)

#make the face detection a LOOP 

cv2.imshow("Face Detector", img)
cv2.waitKey()

'''

###########
print("Code Completed") 