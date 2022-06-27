import cv2
import os 
import time
import sys





path = "/home/andrew/Pictures"
files = os.listdir(path)

index = 0

start = time.time()

print(start)

for file in files:

    img = cv2.imread(path+"/"+file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(path+"/"+file+str(index)+"python.jpg", img)

end = time.time()

print(end)

total = end - start

print(total)





