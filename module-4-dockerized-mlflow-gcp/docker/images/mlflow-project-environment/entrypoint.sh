#! /bin/bash
# add non-root user with user ID that matches user on host
# this allows files saved to bind mounts to be opened and updated 
# without having to change the permissions after running experiments
adduser --disabled-password --gecos "" --uid ${USER_UID} mlflow

# directory added when running project with "mlflow run" command
mlflow_dir=/mlflow/projects/code
if [ -d ${mlflow_dir} ]; # container built by "mlflow run"
then
    # run in the mlflow project directory
    workdir=${mlflow_dir}
    chown -R mlflow ${workdir}
else # container executed manually by user
    # run in home directory
    workdir=/home/mlfow
fi

# run project as non-root user 
# with same UID as user who called mlflow run on host machine
# more info on why we need su-exec here 
# (they use gosu, which does the same thing but is larger/written in go instead of c):
# https://denibertovic.com/posts/handling-permissions-with-docker-volumes/
cd ${workdir} && export USER=mlflow PATH && \
	exec /usr/local/sbin/su-exec mlflow "$@"
