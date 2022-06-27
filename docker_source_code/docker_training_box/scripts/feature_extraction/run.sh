


cat /scripts/feature_extraction/art.txt
cat /scripts/feature_extraction/banner.txt

echo ${DATASET_NAME}
echo ${LABEL_NAME}
echo ${PATH_TO_VIDEOS}
echo ${MODEL}

python -u /scripts/feature_extraction/segment.py -i /videos/${PATH_TO_VIDEOS} -m /models/${MODEL} -l ${LABEL_NAME} -n ${DATASET_NAME}
