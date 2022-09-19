FROM nginx:1.23.1-alpine
RUN apk add --no-cache apache2-utils
WORKDIR /home
COPY nginx.conf /etc/nginx/nginx.conf
COPY entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]
