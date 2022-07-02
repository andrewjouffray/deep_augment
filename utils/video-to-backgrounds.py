import cv2
import argparse
import os
import numpy as np
import time
import random

# define input arguments and help text for each argument
def defineArguments():
    parser = argparse.ArgumentParser(description='Takes a directory of videos (AVI / MP4), and saves each frame in a directory of your choice')
    parser.add_argument('-i', '--input', help="Directory where to find all the videos")
    parser.add_argument('-o', '--output', help="Directory where to save all the images")
    args = parser.parse_args()
    return args

# returns a list of absolute paths to the video files
def joinInputFiles(inputPath):
    files = os.listdir(inputPath)
    absolute_files = []
    for file in files:
            joined_paths = os.path.join(inputPath, file)
            absolute_files.append(joined_paths)
    return absolute_files

# checks the file type and only return MP$ or AVI files
def validateInputFiles(files):

    validFiles = []

    for file in files:
        if file.lower().endswith(".avi") or file.lower().endswith(".mp4") or file.lower().endswith(".mov"):
            validFiles.append(file)

    return validFiles


# makes sure the input path contains at least 1 avi or mp4 video, and return a list of compatible paths
def getInputFilePaths(inputPath):

    files = joinInputFiles(inputPath)

    return validateInputFiles(files)

    
# creates an output directory if none are provided
def createOutputDir(outputPath):

    # skip if an exception occured (assuming it is because the path already exists)
    try:
        os.mkdir(outputPath)
    except Exception as e:
        print("Error: ", e)

def current_milli_time():
    return round(time.time() * 1000)

def blur(img):

    kernel_size = random.randint(2, 6)

    blur = cv2.blur(img,(kernel_size,kernel_size))
    return blur



def saveVideoFrames(file, outputPath):

    cap = cv2.VideoCapture(file)

    i = 0

    while True:

        ret, frame = cap.read()

        if frame is None:
            print("Done saving frames of ", file)
            break

        #frame = blur(frame)

        file_name = "background_" + str(current_milli_time()) + ".jpg"
        save_path = os.path.join(outputPath, file_name)

       
        if i % 4 == 0:

            cv2.imwrite(save_path, frame)

        i += 1

    cap.release()
        
def main():

    
    args = defineArguments()

    inputPath = args.input
    outputPath = args.output

    inputFilePaths = getInputFilePaths(inputPath)

    createOutputDir(outputPath)

    for video_file in inputFilePaths:

        saveVideoFrames(video_file, outputPath)




if __name__ == "__main__":
    main()
