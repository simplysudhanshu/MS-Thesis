FROM ubuntu:latest
LABEL tag=btq-v1
SHELL ["/bin/bash", "-c"]
RUN apt-get -y update && apt-get -y upgrade
RUN apt install -y git curl
RUN git clone https://github.com/simplysudhanshu/bits_to_qubits.git
RUN chmod +x init.sh \
    ./init.sh

# docker build -t btq:v1 .
# docker run -d -t --name btq-container btq:v1