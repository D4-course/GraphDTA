# GraphDTA

Team No: 2

Team Members:
- 2020101132, Mugundan Kottur Suresh
- 2021121003, Manav Chaudhary

## Youtube Video Link

![D4 Project Presentation](https://youtu.be/9y807Q4H2zk)

## Problem Statement

The paper we were assigned tests the hypothesis that a graph structure could yield a more accurate binding affinity prediction than current models by repesenting the drugs as a graph in the prediction process.

## Prerequisites

- Have Docker installed on your machine. If you don't have Docker installed, you can find instructions for your operating system [here](https://docs.docker.com/install/).
- Have CUDA installed on your machine. If you don't have CUDA installed, you can find instructions for your operating system [here](https://developer.nvidia.com/cuda-downloads).
- Have Python 3.9 installed on your machine. If you don't have Python 3.9 installed, you can find instructions for your operating system [here](https://www.python.org/downloads/).
- Install streamlit and requests using pip. If you don't have pip installed, you can find instructions for your operating system [here](https://pip.pypa.io/en/stable/installing/).

## How to Run

- Clone this repository
- Navigate to the `Docker Backend` directory of this repository
- Run `docker build -t graphdta .` to build the Docker image
- Run `docker run -d -p 8000:8000 openfold` to create and run a container from the image
- Navigate to the `Frontend` directory and run `streamlit run frontend.py` to run the frontend