FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
WORKDIR /code
COPY ./code/main.py /code
