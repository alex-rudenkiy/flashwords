# syntax=docker/dockerfile:1.2

# Stage 1: Builder
FROM python:3.11-slim AS builder
ENV POETRY_VERSION 1.5.1

# Установка Poetry
RUN pip install "poetry==$POETRY_VERSION"

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем только pyproject.toml и poetry.lock
COPY pyproject.toml poetry.lock vocab.csv ./


# Устанавливаем зависимости
# RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi
RUN #--mount=type=cache,target=/home/.cache/pypoetry/cache \
    #--mount=type=cache,target=/home/.cache/pypoetry/artifacts \
    poetry install --no-interaction --no-ansi

# Копируем оставшиеся файлы проекта
COPY . .

# Stage 2: Final image
FROM python:3.11-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем зависимости и исходники из builder stage
#COPY --from=builder /app/venv /app/venv
COPY --from=builder /app .
COPY --from=builder /app/vocab.csv /app/vocab.csv

# Указываем точку входа
CMD ["python", "-m", "uvicorn", "main:app"]
