#! /bin/bash 
# Enter storage bucket, sql connection string, postgres password, and username
# name of your storage bucket
export BUCKET=
# SQL connection string, <project>:<zone>:<name>
export SQL_CONNECTION=
# password for default account on postgres
export POSTGRES_PASSWD=
# YOUR account (without @gmail.com), so the startup script knows which home directory to use
export GCP_USER=
# Set the username and password for accessing jupyter and the mlflow ui remotely
export LOGIN_USER=
export LOGIN_PASSWD=

#### Don't change anything below here!!!
export WORKDIR=/home/${GCP_USER}/
export ARTIFACT_URI=gs://${BUCKET}/artifacts
export USER_UID=$(id -u ${GCP_USER})

# if code has not already been configured
if [ ! -d ${WORKDIR}/2022-cloud-mlops-mse-workshop ];
then
    # download the repository
    cd ${WORKDIR} && git clone \
        https://github.com/rccohn/2022-cloud-mlops-mse-workshop.git
    # older version of compose does not have -f {file} argument
    # so rename the correct file docker-compose.yaml
    cd ${WORKDIR}/2022-cloud-mlops-mse-workshop/module-4-dockerized-mlflow-gcp/docker/ && \
        mv jupyter-mlflow-ui-compose.yaml docker-compose.yaml
fi

# compose isn't installed by default, so we can use a trick to run it through docker.
# -v /var/run/docker.sock:/var/run/docker.sock allows docker compose in the container
# to control docker on the host
# -e arguments simply pass environment variables from host to container
# -v $(pwd):$(pwd) -w=$(wd) mounts current directory to container and uses it
# as the working directory to run compose
cd ${WORKDIR}/2022-cloud-mlops-mse-workshop/module-4-dockerized-mlflow-gcp/docker && \
    docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
        -e "ARTIFACT_URI" -e "SQL_CONNECTION" -e "POSTGRES_PASSWD" \
	-e "LOGIN_USER" -e "LOGIN_PASSWD" -e "WORKDIR" -e "USER_UID"\
        -v $(pwd):$(pwd) -w=$(pwd) docker/compose:alpine-1.29.2 \
        up 
