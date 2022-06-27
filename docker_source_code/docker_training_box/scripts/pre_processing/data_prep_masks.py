import os
import sys
import random
import subprocess

# get all the input files
# start a k fold loop

# split the files 90% 10%
# trucate length to be divisble by 64
# format to .record
# save 	in folder with fold#

# python dataprep.py <path> <number of folds>

def main():

	path = sys.argv[1]
	folds = sys.argv[2]

	print("FORMAT: loading dataset")
	
	files = os.listdir(path + "images/")

	# make dataset size divisble by 64 and remove file extention
	for idx, item in enumerate(files):

		fname = item[:len(item) - 4]
		files[idx] = fname
		
	print("FORMAT: starting k-fold cross-validation 90% - 10%")

	for fold in range(int(folds)):
		print("FORMAT: fold #"+ str(fold))
		
		lim = int(len(files) / 10)

		print("FORMAT: limit = " + str(lim))		

		val = files[:lim]
		train = files[lim +1:len(files)]
		
		print("FORMAT: validation =", len(val))
		print("FORMAT: training =", len(train))


		files = train + val

		# make a directory for the fold
		fold_path = path+"fold"+str(fold)
		try:
			
			os.mkdir(fold_path)
			os.mkdir(fold_path+"/train_img")
			os.mkdir(fold_path+"/test_img")
			os.mkdir(fold_path+"/annotation")

		except Exception as e:
			print(e)

		# copy all the files in their respective folders
		added_train = 0
		for item in train:

			if len(train) % 64 != 0:
				train.remove(item)

		
		print(len(train))
		for item in train:

			print(item)
			item = item.strip(".")

			added_train +=1
			command = "cp "+path+"images/"+item+".jpg "+fold_path+"/train_img/"
			process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)

			command = "cp "+path+"images/"+item+".json "+fold_path+"/train_img/"
			process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)

		print("Images in Trainign dataset: ",added_train)

		added_val = 0
		for item in val:
			if len(val) % 64 != 0:
				val.remove(item)
		
		for item in val:

			item = item.strip(".")

			added_val +=1	
			command = "cp "+path+"images/"+item+".jpg "+fold_path+"/test_img/"
			process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)

			command = "cp "+path+"images/"+item+".json "+fold_path+"/test_img/"
			process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)



if __name__ == "__main__":

	main()
