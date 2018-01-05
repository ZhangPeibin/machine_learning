#!/usr/bin/env python
# coding=utf-8
import cv2
import os
from dataset import Model, model_path


def getVideo(window_name, camera_id, catch_pic_num, path_name):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    model = Model()
    model.load_model(model_path)

    # 人脸识别分类器
    classfier = \
        cv2.CascadeClassifier("/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml")

    # 识别出人脸后要画的边框的颜色,RGB格式
    color = (0, 255, 0)

    # open video
    capture = cv2.VideoCapture(camera_id)

    captureNum = 0

    if not os.path.exists("%s/" % path_name):
        os.makedirs("%s/" % path_name)

    while capture.isOpened():
        ok, img = capture.read()

        # 转为灰度，容易识别
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        if len(faceRects) > 0:
            for rect in faceRects:
                x, y, w, h, = rect
                img_name = '%s/me_%d.jpg' % (path_name, captureNum)
                image = img[y - 10:y + h + 10, x - 10:x + w + 10]
                #cv2.imwrite(img_name, image)
                faceId = model.face_predict(image)
                print faceId

                captureNum += 1
                if captureNum > catch_pic_num:
                    break

                cv2.rectangle(img, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, 'num:%d' %
                            (captureNum), (x + 30, y + 30), font, 1, (255, 0, 255), 4)

        if captureNum > catch_pic_num:
            break

        cv2.imshow(window_name, img)
        key = cv2.waitKey(1)
        if key == 27:
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    getVideo("截取视频流", 0, 3000, 'data/me')
