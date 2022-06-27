import os
import sys
import random
import shortuuid
import json

path = sys.argv[1]

image_path = os.path.join(path, "images")

# do not iterate over every file!! only images
for file in os.listdir(image_path):

    if file.endswith("jpg"):
	
        name = os.path.splitext(file)[0]

        new_name = str(random.randint(1, 10)) + str(shortuuid.uuid()) + str(random.randint(1, 1000))

        json_name = name + ".json"
        img_name = name + ".jpg"
        
        new_json_name = new_name + ".json"
        new_img_name = new_name + ".jpg"

        json_file = os.path.join(image_path, json_name)
        img_file = os.path.join(image_path, img_name)

        new_json_file = os.path.join(image_path, new_json_name)
        new_img_file = os.path.join(image_path, new_img_name)

        try:

            os.rename(json_file, new_json_file)
            os.rename(img_file, new_img_file)

            with open(new_json_file, "r") as jsonFile:
                data = json.load(jsonFile)

            data["imagePath"] = new_img_name

            with open(new_json_file, "w") as jsonFile:
                json.dump(data, jsonFile)

            print(name)

        except Exception as e:
            print(e)


