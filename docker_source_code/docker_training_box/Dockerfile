FROM "tensorflow/tensorflow:latest-gpu"

RUN apt-get update && yes | apt-get upgrade

RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN mkdir -p /tensorflow/models

RUN apt-get install -y git python3-pip

RUN apt-get install -y vim 

RUN pip3 install --upgrade pip

RUN pip3 install tensorflow

RUN apt-get install -y protobuf-compiler python-pil python-lxml

RUN pip3 install matplotlib setuptools opencv-python labelme contextlib2 pycocotools lvis scipy tensorflow_io tf-models-official shortuuid secure-smtplib

RUN pip3 install tf_slim shapely

RUN git clone https://github.com/tensorflow/models.git /tensorflow/models

WORKDIR /tensorflow/models/research

RUN protoc object_detection/protos/*.proto --python_out=.

ENV PYTHONPATH "${PYTHONPATH}/tensorflow/models/research:/tensorflow/models/research/slim"

WORKDIR /

COPY ./scripts /scripts

RUN chmod +x /scripts/start.sh


