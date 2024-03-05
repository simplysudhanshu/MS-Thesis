FROM ubuntu:latest
LABEL tag=btq-v1
SHELL ["/bin/bash", "-i", "-c"]
COPY init.sh init.sh
RUN apt-get -y update && apt-get -y upgrade
RUN apt install -y git curl
RUN chmod +x init.sh && ./init.sh 2>&1 | tee init_logs

# ENTRYPOINT [ "git", "clone",  "https://github.com/simplysudhanshu/bits_to_qubits.git" ]

# docker build --no-cache --pull -t btq:v1 .
# docker run -d -t --name btq-container btq:v1