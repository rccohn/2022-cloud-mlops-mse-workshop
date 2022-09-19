#! /bin/ash
# create password file
htpasswd -cb /etc/nginx/.htpasswd ${LOGIN_USER} ${LOGIN_PASSWD}

# start nginx without daemon
nginx -g "daemon off;"
