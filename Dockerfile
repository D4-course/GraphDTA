FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu18.04

# metainformation
LABEL org.opencontainers.image.version = "1.0.0"
LABEL org.opencontainers.image.authors = "Gustaf Ahdritz"
LABEL org.opencontainers.image.source = "https://github.com/aqlaboratory/openfold"
LABEL org.opencontainers.image.licenses = "Apache License 2.0"
LABEL org.opencontainers.image.base.name="docker.io/nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04"

RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get update && apt-get install -y wget curl unzip libxml2 cuda-minimal-build-11-3 libcusparse-dev-11-3 libcublas-dev-11-3 libcusolver-dev-11-3 vim gcc python3.8 python3.8-dev python3-pip

RUN python3.8 -m pip install --upgrade pip

RUN python3.8 -m pip install pillow==6.2.2

RUN python3.8 -m pip install torch==1.11.0+cpu torchvision==0.12.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu

RUN python3.8 -m pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cpu.html

RUN python3.8 -m pip uninstall -y torch-geometric

RUN python3.8 -m pip install torch-geometry==1.7.2

RUN python3.8 -m pip install numpy pandas networkx rdkit

RUN mkdir /GraphDTA

RUN cd /GraphDTA

COPY graphdta.py /GraphDTA/main.py

COPY predict_with_pretrained_model.py /GraphDTA/predict_with_pretrained_model.py

COPY utils.py /GraphDTA/utils.py

COPY create_data.py /GraphDTA/create_data.py

COPY trained_models /GraphDTA/trained_models

COPY models /GraphDTA/models

WORKDIR /GraphDTA

CMD python3 /GraphDTA/main.py