# FROM registry.codeocean.com/codeocean/ubuntu:20.04.2
FROM registry.codeocean.com/codeocean/miniconda3:4.12.0-python3.9-ubuntu20.04

ENV TZ=Europe/Rome

COPY ./requirements.txt /tmp/requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r /tmp/requirements.txt

WORKDIR /code