import os
import sys
import matplotlib.pyplot as plt
import collections
import locale

name = sys.argv[1]
path = sys.argv[2]

iou_general = {}
pf_general = {}

top_pf = 0
top_pf_trunc = ""
top_pf_ckpt = ""

top_iou = 0
top_iou_trunc = ""
top_iou_ckpt = ""
original_dataset_size = 299000

# gather all the accuracy data
for dir_name in os.listdir(path):


	if dir_name.startswith(name) and not "zip" in dir_name:
		

		# find the results 
		IoU_filename = os.path.join(path, dir_name, "fold0", "IoU_accuracy.txt")
		PF_filename = os.path.join(path, dir_name, "fold0", "PF_accuracy.txt")
		
		img_count = 0		

		if not "remaining" in dir_name:
			try:
				truncation = int(dir_name[-1])
				img_count = int(((10 - truncation) / 10) * original_dataset_size)
			except:
				img_count = original_dataset_size


		
		else:
			img_count = int(dir_name[-3:])
		
		
		str_size = locale.format("%d", img_count, grouping=True)
		d_name = str_size+" image dataset"

		iou = []
		pf = []

		with open(IoU_filename) as file:
			lines = file.readlines()
			lines = [line.rstrip() for line in lines]
			for line in lines:
				line = line.split(":")
				ckpt = line[0]
				iou_acc = float(line[1])
				iou_acc = round(iou_acc, 2)
				iou.append(iou_acc)

				if iou_acc > top_iou:
					top_iou = iou_acc
					top_iou_ckpt = len(iou)
					top_iou_trunc = d_name
				
		with open(PF_filename) as file:
			lines = file.readlines()
			lines = [line.rstrip() for line in lines]
			for line in lines:
				line = line.split(":")
				ckpt = line[0]
				pf_acc = float(line[1])
				pf_acc = round(pf_acc, 2)
				pf.append(pf_acc)

				if pf_acc > top_pf:
					top_pf = pf_acc
					top_pf_ckpt = len(pf)
					top_pf_trunc = d_name
	

		iou_general[img_count] = iou
		pf_general[img_count] = pf



# order the dict		
od_iou_general = collections.OrderedDict(sorted(iou_general.items(), reverse = True))
od_pf_general = collections.OrderedDict(sorted(pf_general.items(), reverse = True))

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(30, 10))

ax[0].set_ylim([0, 100])
ax[1].set_ylim([0, 100])

for key in od_iou_general:

	x = range(len(od_iou_general[key]))
	y = od_iou_general[key]

	str_size = locale.format("%d", key, grouping=True)

	ax[0].plot(x,y, label=str_size)

	ax[0].legend()

	ax[0].set_title("Intersection over Union accuracy")
	

for key in od_pf_general:

	x = range(len(od_pf_general[key]))
	y = od_pf_general[key]
	
	str_size = locale.format("%d", key, grouping=True)

	ax[1].plot(x,y, label=str_size)

	ax[1].legend()

	ax[1].set_title("Pass / Fail accuracy, (60+ % IoU accuracy to pass)")

plt.savefig('/workdir/testing/results.png')

print("==================== Top acuracy detected ====================")

print("IoU top: ", str(top_iou)+ "% accuracy at ckpt-", top_iou_ckpt, "in the", top_iou_trunc, "dataset")
print("PF top: ", str(top_pf)+ "% accuracy at ckpt-", top_pf_ckpt, "in the", top_pf_trunc, "dataset")

