import os
import sys
import random
import shortuuid
import xml.etree.ElementTree as ET

path = sys.argv[1]

image_path = os.path.join(path, "images")
xml_path = os.path.join(path, "xml")

for file in os.listdir(image_path):
	
	name = os.path.splitext(file)[0]

	new_name = str(random.randint(1, 10)) + str(shortuuid.uuid()) + str(random.randint(1, 1000))

	xml_name = name + ".xml"
	img_name = name + ".jpg"
	
	new_xml_name = new_name + ".xml"
	new_img_name = new_name + ".jpg"

	xml_file = os.path.join(xml_path, xml_name)
	img_file = os.path.join(image_path, img_name)

	new_xml_file = os.path.join(xml_path, new_xml_name)
	new_img_file = os.path.join(image_path, new_img_name)

	os.rename(xml_file, new_xml_file)
	os.rename(img_file, new_img_file)

	print("filename:", new_img_file)

	mytree = ET.parse(new_xml_file)
	myroot = mytree.getroot()
	p = mytree.find("path")
	p.text = new_img_file
	n = mytree.find("filename")
	n.text = new_img_name
	mytree.write(new_xml_file)
	


