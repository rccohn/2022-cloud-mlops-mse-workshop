#! /bin/bash
mlflow server --backend-store-uri sqlite:///mlflow-data/backend.db --artifacts-destination file:mlflow-data --serve-artifacts
