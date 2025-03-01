#!/bin/bash
mkdir ssl
openssl genrsa -des3 -passout pass:qwerty -out server.pass.key 2048
openssl rsa -passin pass:qwerty -in server.pass.key -out ssl/nginx.key
rm server.pass.key
openssl req -new -key ssl/nginx.key -out server.csr -subj "/C=RU/ST=Moscow/L=Moscow/O=YANNIT/OU=IT Department/CN=flashwords.ydns.eu"
openssl x509 -req -days 365 -in server.csr -signkey ssl/nginx.key -out ssl/nginx-certificate.crt
