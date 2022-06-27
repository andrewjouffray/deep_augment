#!/bin/bash

if [ ${PREPROCESSING} = "TRUE" ]
then

	python /scripts/pre_processing/shuffle.py /workdir/input_data
	python /scripts/pre_processing/data_prep_masks.py /workdir/input_data/ 1
	python /scripts/pre_processing/labelMe2Coco.py --labelme_images=/workdir/input_data/fold0/train_img/ --output=/workdir//input_data/fold0/train.json
	python /scripts/pre_processing/labelMe2Coco.py --labelme_images=/workdir/input_data/fold0/test_img/ --output=/workdir//input_data/fold0/test.json
	python /scripts/pre_processing/mask_tf_record_from_json.py --logtostderr --train_image_dir=/workdir/input_data/fold0/train_img --test_image_dir=/workdir/input_data/fold0/test_img/ --train_annotations_file=/workdir/input_data/fold0/train.json --test_annotations_file=/workdir/input_data/fold0/test.json --include_masks=True --output_dir=/workdir/input_data/

fi

if [ ${EXPORT_ONLY} = "TRUE" ]
then
	echo ""
	echo "XXXXXXXXXXXXXXXXXXXXXXXXX EXPORTING XXXXXXXXXXXXXXXXXXXXXXXXXXXX"
	echo ""

	python /scripts/post_processing/export.py /workdir/pre_trained_model/ 0
else

	echo ""
	echo "XXXXXXXXXXXXXXXXXXXXXXXXX TRAINING XXXXXXXXXXXXXXXXXXXXXXXXXXX"
	echo ""


	python /scripts/training_scripts/train_latest.py --model_dir=/workdir/pre_trained_model/training/ --pipeline_config_path=/workdir/pre_trained_model/pipeline.config &

	echo "------------------ saving checkpoints ------------------------"

	python /scripts/training_scripts/saveCheckpoint.py /workdir/pre_trained_model/ &
	
	echo "------------------ waiting for model to train ---------------"

	sleep ${TIMEOUT} ; 

	echo "------------------ killing training and model saving --------"
	
	kill $!

	echo ""
	echo "XXXXXXXXXXXXXXXXXXXXXXXXX EXPORTING XXXXXXXXXXXXXXXXXXXXXXXXXXX"
	echo ""

	python /scripts/post_processing/export.py /workdir/pre_trained_model/ 0
fi


