FROM tensorflow/tensorflow:latest-py3
LABEL maintainer="karaage0703 <coldchicken@gmail.com>"

RUN apt-get update
RUN apt-get -y install git
RUN apt-get -y install wget
RUN apt-get -y install unzip
RUN rm -rf /var/lib/apt/lists/*

RUN pip install matplotlib
RUN pip install pillow
RUN pip install Cython
RUN pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"

RUN git clone --depth 1 https://github.com/tensorflow/models
WORKDIR models/research

RUN wget -O protobuf.zip https://github.com/protocolbuffers/protobuf/releases/download/v3.8.0-rc1/protoc-3.8.0-rc-1-linux-x86_64.zip
RUN unzip -o protobuf.zip -d /usr/local bin/protoc
RUN rm -f protobuf.zip
RUN /usr/local/bin/protoc object_detection/protos/*.proto --python_out=.

RUN git clone https://github.com/karaage0703/object_detection_tools