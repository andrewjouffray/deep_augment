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

# python dataprep.py <path> <trucate n time (10% each time)> <make tf records? (true / false)>

def main():

	path = sys.argv[1]
	truncate = sys.argv[2]
	record = sys.argv[3]

	print("FORMAT: loading dataset")
	
	files = os.listdir(path + "images/")

	# make dataset size divisble by 64 and remove file extention
	for idx, item in enumerate(files):

		fname = item[:len(item) - 4]
		files[idx] = fname
		
	for trunc in range(int(truncate)):

		trunc = trunc / 10

		print("FORMAT: trucating "+ str(trunc) + "%")

		files_to_remove = int(len(files) * trunc)

		limit = len(files) - files_to_remove

		new_files = files[:limit -1]

		
		lim = int(len(new_files) / 10)

		print("FORMAT: limit = " + str(lim))		

		val = new_files[:lim]
		train = new_files[lim +1:len(files)]
		
		print("FORMAT: validation =", len(val))
		print("FORMAT: training =", len(train))


		# make a directory for the fold
		fold_path = path+"trunc"+str(trunc)
		try:
			
			os.mkdir(fold_path)
			os.mkdir(fold_path+"/train_img")
			os.mkdir(fold_path+"/train_xml")
			os.mkdir(fold_path+"/test_img")
			os.mkdir(fold_path+"/test_xml")
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

			item = item.strip(".")
			
			added_train +=1
			command = "cp "+path+"images/"+item+".jpg "+fold_path+"/train_img/"
			process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)


			command = "cp "+path+"xml/"+item+".xml "+fold_path+"/train_xml/"
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


			command = "cp "+path+"xml/"+item+".xml "+fold_path+"/test_xml/"
			process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)


		if record.lower() == "true":

			# send the commands to format the data into train and test record files 
			command = "python xml_record.py -x "+ fold_path +"/train_xml -l "+path+"label.pbtxt -o "+fold_path+"/annotation/train.record -i "+fold_path+"/train_img"

			process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
			output, error = process.communicate()
			
			print(output)
			print(error)

		

			command = "python xml_record.py -x "+ fold_path +"/test_xml -l "+path+"label.pbtxt -o "+fold_path+"/annotation/test.record -i "+fold_path+"/test_img"

			process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
			output, error = process.communicate()

			print(output)
			print(error)

			# re-insert the test images at the end of the training images



if __name__ == "__main__":

	main()
