FROM python:3.10.7-slim-buster
RUN useradd -m mlflow
USER mlflow
WORKDIR /home/mlflow
RUN python -m venv env && ./env/bin/python -m \
        pip install google-cloud-storage==2.5.0 mlflow==1.28.0 psycopg2-binary==2.9.3
ENV PATH=/home/mlflow/env/bin:${PATH}

ENTRYPOINT ["mlflow"]
