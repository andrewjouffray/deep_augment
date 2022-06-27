echo ${DATASET_NAME}
echo ${LABEL_NAME}
echo ${PATH_TO_VIDEOS}


for MODEL in /models/*/
do
	echo "Testing model ${MODEL}"

	python -u /scripts/feature_extraction/segment.py -i /videos/${PATH_TO_VIDEOS} -m ${MODEL} -l ${LABEL_NAME} -n ${DATASET_NAME}
done


