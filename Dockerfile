FROM ubuntu:latest
LABEL tag=btq-v1
SHELL ["/bin/bash", "-i", "-c"]
COPY init.sh init.sh
RUN apt-get -y update && apt-get -y upgrade
RUN apt install -y git curl
RUN chmod +x init.sh && ./init.sh 2>&1 | tee init_logs

## build
# docker build --no-cache --pull -t btq:v1 .

## Run a new container
# docker run -d -t --name btq-container --gpus all btq:v1

## Pause a container
# docker pause btq-container

## Re-start an exited container with changes intact
# docker start btq-container
