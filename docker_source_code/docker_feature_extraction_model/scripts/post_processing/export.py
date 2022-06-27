import os 
import sys
import subprocess

# python export <path to all the auto_* files> <fold number>

path = sys.argv[1]
fold = sys.argv[2]

dirs = path.split("/")

name = dirs[-2]
name = name.strip("/")

files = os.listdir(path)

if not path.endswith("/"):
	path = path+"/"

print(dirs)
print(name)


# make the output directories, ignore the errors if they already exist
try:

	os.mkdir("/home/ajouffray/TF-Object-Detection/exported/"+name)
	os.mkdir("/home/ajouffray/TF-Object-Detection/exported/"+name+"/fold"+fold+"/")
except:
	pass


for file in files:

	if file.startswith("auto"):

		try:
			
			command = "python exporter_main_v2.py --input_type image_tensor --pipeline_config_path "+path+"pipeline.config --trained_checkpoint_dir "+path+file+"/ --output_directory /home/ajouffray/TF-Object-Detection/exported/"+name+"/fold"+fold+"/"+file

			print("command: ", command)

			process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
			output, error = process.communicate()

			print(output)
			print(error)
		
		except Exception as e:
			
			print(e)

		
