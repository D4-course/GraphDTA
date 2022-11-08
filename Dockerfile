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

RUN apt-get update && apt-get install -y wget curl unzip libxml2 cuda-minimal-build-11-3 libcusparse-dev-11-3 libcublas-dev-11-3 libcusolver-dev-11-3 vim gcc python3-dev

# RUN wget -P /tmp \
#     "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" \
#     && bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
#     && rm /tmp/Miniconda3-latest-Linux-x86_64.sh
# ENV PATH /opt/conda/bin:$PATH

# COPY environment.yml /opt/graphdta/environment.yml

# installing into the base environment since the docker container wont do anything other than run GraphDTA
# RUN conda env update -n base --file /opt/graphdta/environment.yml && conda clean --all

# RUN conda install -y -c conda-forge rdkit

# RUN conda install pytorch torchvision cudatoolkit -c pytorch

# RUN conda install pyg -c pyg

# RUN pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.4.0.html

# RUN pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.4.0.html

# RUN pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.4.0.html

# RUN pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.4.0.html

RUN pip install torch-geometric numpy pandas networkx

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