#! /bin/bash

# if container hasn't configured (first time it is started)
if [ ! -d /home/jovyan ];
then
    # if USER_UID isn't specified, use default value UID=1000
    if [ -z ${USER_UID} ];
    then
        echo "USER_UID not defined, using default value 1000 for UID"
	    export USER_UID=1000
    fi
    # add non-root user jovyan
    adduser --disabled-password --gecos "" --uid ${USER_UID} jovyan
    # use default jupyter settings (dark theme, line numbers, etc)
    mkdir -p /home/jovyan/.jupyter/lab
    mv /home/jupyter-settings /home/jovyan/.jupyter/lab/user-settings
    chown -R jovyan /home/jovyan/.jupyter/
    export HOME=/home/jovyan/
fi

# make sure non-root user can access bind-mounted files
chown jovyan /mnt/host-files
export JUPYTER_PASS_HASH=$(python -c "from notebook.auth import passwd; \
    print(passwd('${LOGIN_PASSWD}'))")
exec /usr/local/sbin/su-exec jovyan "python -m jupyterlab --no-browser --ip 0.0.0.0 \
     --no-browser --ServerApp.token= --ServerApp.password=${JUPYTER_PASS_HASH}"
