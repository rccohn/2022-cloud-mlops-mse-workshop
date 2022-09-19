#! /bin/bash 
# Enter storage bucket, sql connection string, postgres password, and username
export ARTIFACT_URI=
export SQL_CONNECTION=
export POSTGRES_PASSWD=
export GCP_USER=

#### Don't change anything below here!!!
export WORKDIR=/home/${GCP_USER}/
# if code has not already been 
if [ ! -d ${WORKDIR}/2022-cloud-mlops-mse-workshop ];
then
    echo "cloning repository"
    cd ${WORKDIR} && git clone \
        https://github.com/rccohn/2022-cloud-mlops-mse-workshop.git
        
    cd ${WORKDIR}/2022-cloud-mlops-mse-workshop/module-4-dockerized-mlflow-gcp/docker/ && \
        mv jupyter-mlflow-ui-compose docker-compose.yaml
fi


cd ${WORKDIR}/2022-cloud-mlops-mse-workshop/module-4-dockerized-mlflow-gcp/docker && \
    docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
        -e "ARTIFACT_URI" -e "SQL_CONNECTION" -e "POSTGRES_PASSWD" \
        -v $(pwd):$(pwd) -w=$(pwd) docker/compose:alpine-1.29.2 \
        up -d
