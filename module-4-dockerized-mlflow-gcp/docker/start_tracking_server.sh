#! /bin/bash 
# Enter storage bucket, sql connection string, postgres password, and username
export ARTIFACT_URI=
export SQL_CONNECTION=
export POSTGRES_PASSWD=
export GCP_USER=

#### Don't change anything below here!!!
export WORKDIR=/home/${GCP_USER}/

# if code has not already been configured
if [ ! -d ${WORKDIR}/2022-cloud-mlops-mse-workshop ];
then
    # pull code from git in Docker
    sudo docker run -v ${WORKDIR}:/mnt --rm alpine:3.16.2 ash -c \
        "cd /mnt && apk add --no-cache git && git clone \
        https://github.com/rccohn/2022-cloud-mlops-mse-workshop.git"
    # older version of compose doesn't have -f file argument, so
    # rename the appropriate config "docker-compose.yaml"
    cd ${WORKDIR}/2022-cloud-mlops-mse-workshop/module-4-dockerized-mlflow-gcp/docker/ && \
        mv mlflow-server-compose.yaml docker-compose.yaml
fi

# add linux firewall rule opening tcp port 5000
sudo iptables -A INPUT -p tcp --dport 5000 -j ACCEPT

# COS doesn't actually have compose, so we run it through docker itself
# start the containers
# -v /var/run/docker.sock:/var/run/docker.sock allows docker compose in the container
# to control docker on the host
# -e arguments simply pass environment variables from host to container
# -v $(pwd):$(pwd) -w=$(wd) mounts current directory to container and uses it
# as the working directory to run compose

cd ${WORKDIR}/2022-cloud-mlops-mse-workshop/module-4-dockerized-mlflow-gcp/docker && \
    docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
        -e "ARTIFACT_URI" -e "SQL_CONNECTION" -e "POSTGRES_PASSWD" \
        -v $(pwd):$(pwd) -w=$(pwd) docker/compose:alpine-1.29.2 \
        up -d
