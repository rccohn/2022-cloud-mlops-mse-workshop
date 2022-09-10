#! /bin/bash
if [ -z ${USER_UID} ];
then
	export USER_UID=1000
fi

adduser --disabled-password --gecos "" --uid ${USER_UID} potts

chown -R potts /mnt/outputs

exec /usr/local/sbin/su-exec potts "$@"
