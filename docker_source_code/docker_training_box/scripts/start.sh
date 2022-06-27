#!/bin/bash

if [ ${PREPROCESSING} = "TRUE" ]
then

	python /scripts/pre_processing/shuffle.py /workdir/input_data/
	echo "--------------------- copying files over and generating .record files, please wait ------------------------------"
	python /scripts/pre_processing/data_prep_boxes.py /workdir/input_data/ 1 true

fi


if [ ${TRAINING} = "TRUE" ]
then
	echo ""
	echo "XXXXXXXXXXXXXXXXXXXXXXXXX TRAINING XXXXXXXXXXXXXXXXXXXXXXXXXXX"
	echo ""


	python /scripts/training_scripts/model_main_tf2.py --model_dir=/workdir/pre_trained_model/training/ --pipeline_config_path=/workdir/pre_trained_model/pipeline.config &

	echo "------------------ saving checkpoints ------------------------"

	python /scripts/training_scripts/saveCheckpoint.py /workdir/pre_trained_model/ &
	
	echo "------------------ waiting for model to train ---------------"

	sleep ${TIMEOUT} ; 

	echo "------------------ killing training and model saving --------"
	
	kill $!

fi


if [ ${EXPORT} = "TRUE" ]
then
	echo ""
	echo "XXXXXXXXXXXXXXXXXXXXXXXXX EXPORTING XXXXXXXXXXXXXXXXXXXXXXXXXXXX"
	echo ""

	python /scripts/post_processing/export.py /workdir/pre_trained_model/ 0
fi


if [ ${TEST} = "TRUE" ]
then
	echo ""
	echo "XXXXXXXXXXXXXXXXXXXXXXXXX TESTING XXXXXXXXXXXXXXXXXXXXXXXXXXXX"
	echo ""

	python /scripts/testing_scripts/box_test_model2.py /workdir/test_input/imgs /workdir/test_input/xml /workdir/exported/pre_trained_model/fold0/ /workdir/input_data/label.pbtxt 0
	python /scripts/testing_scripts/collect_results.py pre /workdir/testing/
	python /scripts/testing_scripts/email_results.py
fi



