import cv2
import os
import numpy as np
datasets='datasets'
haarcascade='haarcascade_frontalface_default.xml'
(images,labels,names,id)=([],[],{},0)
for (root,dirs,files)in os.walk(datasets):
    for subdir in dirs:
        names[id]=subdir
        subjectpath=os.path.join(root,subdir)
        for filename in os.listdir(subjectpath):
            path=os.path.join(subjectpath,filename)
            label=id
            images.append(cv2.imread(path,0))
            labels.append(int(label))
        id=id+1
(images,labels)=[np.array(lis) for lis in [images,labels]]
model=cv2.face.FisherFaceRecognizer_create()
model.train(images,labels)
facecascade=cv2.CascadeClassifier(haarcascade)
cnt=0
vs=cv2.VideoCapture(0)
while True:
    a,img=vs.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=facecascade.detectMultiScale(gray,1.3,3)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        face=gray[y:y+h,x:x+w]
        face_resize=cv2.resize(face,(130,100))
        prediction=model.predict(face_resize)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        if prediction[1]<800:
            cv2.putText(img,'%s-%.0f'% (names[prediction[0]],prediction[1]),(x-10, y-10), cv2.FONT_HERSHEY_COMPLEX,1,(51, 255, 255))
            print(names[prediction[0]])
            cnt=0
        else:
            cnt=cnt+1
            cv2.putText(img,'unknown',(x-10,y-10),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0))
            if(cnt>100):
                print('UNKNOWN')
                cv2.imwrite('input.jpg',img)
                cnt=0
    cv2.imshow('show',img)
    key=cv2.waitKey(10)
    if key==27:
        break
vs.release()
cv2.destroyAllWindows()

                
                
    
            
