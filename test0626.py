import cv2
from matplotlib import pyplot as plt
import numpy as np
import math

camera_id = 0
delay = 1
window_name = 'frame'

cap = cv2.VideoCapture(camera_id)

if not cap.isOpened():
    print("映像が取得できません。")

    import sys
    sys.exit()

while True:
    ret, frame = cap.read()
    if frame is None:
        break


    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #blur = cv2.GaussianBlur(gray, (0, 0), 5)
    #画像の平滑化
    kernel = np.ones((5,5),np.float32)/25
    dst = cv2.filter2D(frame,-1,kernel)
    #輪郭抽出
    canny_img = cv2.Canny(dst, 50, 110)

    #円の検出
    circles = cv2.HoughCircles(canny_img, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=100, param2=40, minRadius=10, maxRadius=50)
    circles = np.uint16(np.around(circles))

    for circle in circles[0, :]:
        # 円周を描画する
        cv2.circle(frame, (circle[0], circle[1]), circle[2], (0, 255, 0), 5)
        # 中心点を描画する
        cv2.circle(frame, (circle[0], circle[1]), 2, (0, 0, 255), 3)

        print([circle[0],circle[1],circle[2]])
        print(circle)
    
    
    cv2.imshow(window_name, frame)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cv2.destroyWindow(window_name)

