#! /bin/bash
# this an example for the process for running mlflow project
# for simple projects like the one in this workshop, it
# is typically easier to just run the commands manually

# load tracking uri environment variable
source .env

# get UID to pass to nonroot container 
export USER_UID=$(id -u $(whoami))

# invoke mlflow run
source activate env && mlflow run 


