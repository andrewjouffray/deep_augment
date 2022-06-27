import os 
import cv2

files = os.listdir("../uncompiled_random_shapes/")

print(files)

out = cv2.VideoWriter('../compiled_shapes.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (500,500))
for file in files:

    img = cv2.imread("../uncompiled_random_shapes/"+file)

    print("processing file: " + "../uncompiled_random_shapes/"+file)

    out.write(img)

out.release()
