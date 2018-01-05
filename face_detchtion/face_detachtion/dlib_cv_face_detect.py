#!/usr/bin/env python
#coding = utf-8
import cv2
import os
import dlib

#use dlib face detector
detector = dlib.get_frontal_face_detector()


def detect(img,num,path_name):
    greyimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    rects = detector(greyimg,1)
    color = (0,255,0)
    if len(rects) > 0:
        for rect in rects:
            l,t,r,b = rect.left(),rect.top(),rect.right(),rect.bottom()
            captureImg = img[l-20:r+20,t-20:b+20]
            img_name = '%s/%d.jpg' %(path_name,num)
            cv2.imwrite(img_name,captureImg)
            cv2.rectangle(img,(l-10,t-10),(r+10,b+10),color,2)

        return True
    else:
        return False


def openVideo(window_name,path_name,max_pic=100,camera_id=0):

    if not os.path.exists("%s/"%path_name):
        os.makedir("%s/"%path_name)

    cv2.namedWindow(window_name)
    capture = cv2.VideoCapture(camera_id)

    num=0
    while capture.isOpened():
        ok,img = capture.read()

        if ok:
            if detect(img,num,path_name):
                num=num+1

        cv2.imshow(window_name,img)
        keycode =  cv2.waitKey(1)
        if keycode == 27:
            break

        if num > max_pic:
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    openVideo("dlib_opencv","data/me")
