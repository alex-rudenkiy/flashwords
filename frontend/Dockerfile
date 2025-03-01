# Stage 1: Builder
FROM node:20-alpine AS builder

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем package.json и yarn.lock
COPY package*.json ./

# Устанавливаем зависимости
RUN yarn install

# Копируем остальной код проекта
COPY . .

# Build the app
RUN yarn build

# RUN apk update && \
#     apk add openssl && \
#     openssl genrsa -des3 -passout pass:x -out server.pass.key 2048 && \
#     openssl rsa -passin pass:x -in server.pass.key -out nginx.key && \
#     rm server.pass.key && \
#     openssl req -new -key nginx.key -out server.csr \
#         -subj "/C=RU/ST=Moscow/L=Moscow/O=YANNIT/OU=IT Department/CN=flashwords.ydns.eu" && \
#     openssl x509 -req -days 365 -in server.csr -signkey nginx.key -out nginx-certificate.crt


# Stage 2: Final image (using nginx as a server)
FROM nginxinc/nginx-unprivileged  

# Copy built app from builder stage
COPY --chown=nginx:nginx --from=builder /app/build /usr/share/nginx/html
# COPY --chown=nginx:nginx --from=builder /app/nginx.key /tmp/nginx/ssl/nginx.key
# COPY --chown=nginx:nginx --from=builder /app/nginx-certificate.crt /tmp/nginx/ssl/nginx-certificate.crt

# Copy nginx config
COPY nginx.conf /etc/nginx/nginx.conf

# Expose port
EXPOSE 8080

# CMD ["sleep", "infinity"]
CMD ["nginx", "-g", "daemon off;"]
