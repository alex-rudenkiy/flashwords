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

# Stage 2: Final image (using nginx as a server)
FROM nginx:alpine

# Copy built app from builder stage
COPY --from=builder /app/build /usr/share/nginx/html

# Copy nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
