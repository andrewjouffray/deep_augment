import os
import sys
import shutil
import time
import subprocess

#python saveCheck.py path/to/saved/checkpoints

def main(args):

	lastCheck = 0;
	
	path = args[1]
	train_path = path+"training"
	print("Checking on path: " + train_path)

	tries = 0

	while True:

		if tries > 1600:
			break

		tries += 1

		try:
			files = os.listdir(train_path)
		except Exception as e:
			print("could not find the directory: " + train_path)
			print("cause:", e)
			exit()

		for file in files:

			try:
				if file.startswith("ckpt"):
					
					splitF = file.split(".")
					name = splitF[0]
					print("checking file ", name)

					number = 0;
					
					number = name[-2:]
					number = int(number)

					if number < 0:
					
						number = name[-1]
						number = int(number)
						print("single digit number ", number)

					print("got number: ", number)

					if number > lastCheck:
						print("saving checkpoint ", number)
						lastCheck = number
						
						os.mkdir(path+"auto_save_ckpt"+str(number))
						command = "cp -r " + train_path + "/ckpt-"+str(number)+".index "+path+"auto_save_ckpt"+str(number)
						print("trying: ", command)
						process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)

						command = "cp -r " + train_path + "/ckpt-"+str(number)+".data-00000-of-00001 "+path+"auto_save_ckpt"+str(number)
						print("trying: ", command)
						process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)

						command = "cp -r " + train_path + "/train "+path+"auto_save_ckpt"+str(number)
						print("trying: ", command)
						process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)

						command = "cp -r " + train_path + "/checkpoint "+path+"auto_save_ckpt"+str(number)
						print("trying: ", command)
						process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)


					else:
						pass

			except Exception as e:

				print("error: ", e)

		time.sleep(60)
		print("checked")


if __name__ == "__main__":

	main(sys.argv)
