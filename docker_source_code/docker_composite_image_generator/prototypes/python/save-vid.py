import cv2
import os 
import time
import sys





path = "/mnt/0493db9e-eabd-406b-bd32-c5d3c85ebb38/Projects/Video/Weeds2/nenuphare/data1595119927.9510028output.avi"
save = "/mnt/0493db9e-eabd-406b-bd32-c5d3c85ebb38/Projects/dump/"


start = time.time()

print(start)

cap = cv2.VideoCapture(path)

index = 0

while True:

    ret, frame = cap.read()
    if ret == False:
        print("all done")
        break

    img = frame
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(save+str(index)+"python.jpg", img)

    index += 1

end = time.time()

print(end)

total = end - start

print(total)





