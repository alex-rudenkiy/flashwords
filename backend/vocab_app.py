import typer
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import random
import json
import os

app = typer.Typer()
data_dir = "user_data"
default_csv = "vocab.csv"


def load_data(user):
    """Загружает данные пользователя из файла."""
    filename = os.path.join(data_dir, f"{user}.json")
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            user_data = json.load(f)
            # Конвертируем last_reviewed обратно в datetime
            for word in user_data:
                if isinstance(word.get("last_reviewed"), str):  # Проверяем, что это строка
                    try:
                       word["last_reviewed"] = datetime.fromisoformat(word["last_reviewed"])
                    except ValueError:
                        word["last_reviewed"] = datetime.min # Обработка некорректных форматов
            return user_data
    return []


def save_data(user, data):
    """Сохраняет данные пользователя в файл."""
    os.makedirs(data_dir, exist_ok=True)
    filename = os.path.join(data_dir, f"{user}.json")
    # Конвертируем datetime в строку
    for word in data:
      if isinstance(word.get("last_reviewed"), datetime):
          word["last_reviewed"] = word["last_reviewed"].isoformat()
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# Умный алгоритм выбора следующего слова
def smart_pick(user_words):
    if not user_words:
      return None

    now = datetime.now()

    eligible_words = [
      word for word in user_words if now - datetime.fromisoformat(str(word["last_reviewed"])) > timedelta(minutes=1)
    ]
    if not eligible_words:
      return None

    # Вычисляем приоритет для каждого слова
    def calculate_priority(word):
      score_value = {
        "не знаю": 0,
        "почти запомнил": 1,
        "помню": 2,
        "вызубрил": 3,
      }[word["score"]]

      time_since_review = (now - datetime.fromisoformat(str(word["last_reviewed"]))).total_seconds()
      priority = (4 - score_value) * 10 + time_since_review / 60
      return priority

    # Сортируем слова по приоритету (чем меньше, тем раньше)
    eligible_words.sort(key=calculate_priority)

    return eligible_words[0]


@app.command()
def import_csv(user: str, file: str = default_csv):
    """Импортировать слова из CSV-файла."""
    try:
        df = pd.read_csv(file)
        user_data = load_data(user)  # Загружаем существующие данные
        for _, row in df.iterrows():
            user_data.append({
                "id": int(row["id"]),
                "foreignWord": str(row["иностранное слово"]),
                "nativeWord": str(row["слово на языке носителя"]),
                "description": str(row["описание"]),
                "score": "не знаю",
                "last_reviewed": datetime.min
            })
        save_data(user, user_data)
        typer.echo(f"Словарь пользователя {user} успешно обновлён.")
    except FileNotFoundError:
        typer.echo(f"Ошибка: Файл {file} не найден.")
    except pd.errors.EmptyDataError:
        typer.echo("Ошибка: CSV файл пуст.")
    except Exception as e:
        typer.echo(f"Непредвиденная ошибка: {e}")


@app.command()
def edit_word(user: str, word_id: int, foreign: Optional[str] = None, native: Optional[str] = None, desc: Optional[str] = None):
    """Редактировать слово в словаре."""
    user_data = load_data(user)
    for word in user_data:
        if word["id"] == word_id:
            if foreign:
                word["foreignWord"] = foreign
            if native:
                word["nativeWord"] = native
            if desc:
                word["description"] = desc
            save_data(user, user_data)
            typer.echo("Слово успешно обновлено.")
            return
    typer.echo("Слово не найдено.")


@app.command()
def learn(user: str):
    """Начать тренировку слов."""
    user_data = load_data(user)
    if not user_data:
        typer.echo("Словарь пользователя пуст. Импортируйте слова.")
        return

    while True:
        word = smart_pick(user_data)
        if not word:
            typer.echo("Все слова выучены или недоступны для повторения сейчас.")
            break

        typer.echo(f"\n{'—' * 35}")
        typer.echo(f"{word['foreignWord']:^30}")
        typer.echo("[Check]")
        input("Нажмите Enter, чтобы проверить перевод...")

        typer.echo(f"\nПеревод: {word['nativeWord']} ({word['description']})")
        typer.echo("\nВыберите оценку:")
        typer.echo(" (1) Не знаю")
        typer.echo(" (2) Почти запомнил")
        typer.echo(" (3) Помню")
        typer.echo(" (4) Вызубрил")

        while True:
            choice = input("Ваш выбор (1-4): ")
            if choice in {"1", "2", "3", "4"}:
                scores = ["не знаю", "почти запомнил", "помню", "вызубрил"]
                word["score"] = scores[int(choice) - 1]
                word["last_reviewed"] = datetime.now()
                save_data(user, user_data)
                break
            else:
              typer.echo("Некорректный выбор. Введите число от 1 до 4.")


@app.command()
def export_csv(user: str, file: str = default_csv):
    """Экспортировать словарь в CSV."""
    user_data = load_data(user)
    if not user_data:
        typer.echo("Словарь пользователя пуст.")
        return
    df = pd.DataFrame(user_data)
    df.to_csv(file, index=False)
    typer.echo(f"Словарь пользователя {user} сохранён в {file}.")

if __name__ == "__main__":
    app()
