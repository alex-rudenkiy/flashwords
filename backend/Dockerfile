# syntax=docker/dockerfile:1.2

# Stage 1: Builder
FROM sunpeek/poetry:py3.11-slim as requirements-stage

# # Установка Poetry
# RUN pip install "poetry==$POETRY_VERSION"

# # Устанавливаем рабочую директорию
WORKDIR /app

# # Копируем только pyproject.toml и poetry.lock
# COPY pyproject.toml poetry.lock vocab.csv ./


# (4)
COPY ./pyproject.toml ./poetry.lock* /app/

# (5)
RUN poetry export -f requirements.txt --output requirements.txt 

# (6)
FROM python:3.11-slim

# (7)
WORKDIR /app

# (8)
COPY --from=requirements-stage /app/requirements.txt /app/requirements.txt

# (9)
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# (10)
COPY ./app .

# (11)
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
