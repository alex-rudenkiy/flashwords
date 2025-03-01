import random
import statistics
from collections import defaultdict
import os
from llm import get_sentence
from questions import randomQuestionFactory, simpleQuestionFactory

import casdoor
import numpy as np
from fastapi import FastAPI, HTTPException, Path, Body, Depends, Request
from fastapi_another_jwt_auth import AuthJWT
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from tinydb import TinyDB, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from casdoor import CasdoorSDK
import jwt
import json
import requests

# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
# import bitsandbytes

# from llama_cpp import Llama

# llm = Llama.from_pretrained(
# 	repo_id="bartowski/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF",
# 	filename="Llama-3.1-8B-Lexi-Uncensored-V2-Q8_0.gguf",
# )
#
#
#
#

# # r = llm.create_chat_completion(
# # 	messages = [
# # 		{
# # 			"role": "user",
# # 			"content": "What is the capital of France?"
# # 		}
# # 	]
# # )
#
# output = llm(
#       "Q: "+input_text+" A: ", # Prompt
#       max_tokens=32, # Generate up to 32 tokens, set to None to generate up to the end of the context window
#       stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
#       echo=True # Echo the prompt back in the output
# ) # Generate a completion, can also call create_completion
# print(output)

get_sentence("hello")


public_key = os.environ.get('Casdoor_Certificate', """-----BEGIN CERTIFICATE-----
MIIE3TCCAsWgAwIBAgIDAeJAMA0GCSqGSIb3DQEBCwUAMCgxDjAMBgNVBAoTBWFk
bWluMRYwFAYDVQQDEw1jZXJ0LWJ1aWx0LWluMB4XDTI1MDExMzE0MDM1OVoXDTQ1
MDExMzE0MDM1OVowKDEOMAwGA1UEChMFYWRtaW4xFjAUBgNVBAMTDWNlcnQtYnVp
bHQtaW4wggIiMA0GCSqGSIb3DQEBAQUAA4ICDwAwggIKAoICAQCWaux5rvaSxumz
ghlWFydTVcFkMzkrcCybJGDlBOOKgbscsMxH+dRahH5TZCih2UY/kANWNIlBpvOe
UlRx9OjKIKFDDyvkGLeb7H8UzDgDGy+xpSC5tM+4UL9bka1cJEuc4OWOMFvm/Fd5
XV366aVGEMuvPHTokKyQfMgYVpnmnQRFQWMFV1EIsb3PgzeaLRfJwPg9Nb7KDh0i
uZyAsAYRdVFm3fKI5Z+Bwe8kaFPy92GnJTuV4MBNr484GpD96JbDTLvahr8764zl
wT4BZagdMja9dHVNgW9M4566YISDytjuPSOaFHJmdDaImXUOUna924EsVzvmoil7
9B6IRwJBhghBIQZps2BgoKilsY2jYYvuzQzig4GV52YyzMvIbIPFhT4Dq3pzOe86
+RjZw7G2KnsB/nY90j57JOUFthIaZQb8Wj2R8z1IO3FNY3wnCFhvgDuWQOnIptXi
6bZg7SRPCq/AqSTrHwe+hzH4JKLIQ0qBh1c34+3Fgej4M1n51UtzHHMSVnETDzac
GCXx0ARivNyk8ZXUY07JLmDIhIPl0dzVYZgMLTMt12OEYJKx1irkz2oAlF3jVZei
fvkN5a+CUxABtedToRd8x5rnIUYXCOBZuR/UxRyVO4P0tqm1PoFIuoeBttZcU56P
DSGNaIqqUVwnjDDNoV+lCDiAJai/8QIDAQABoxAwDjAMBgNVHRMBAf8EAjAAMA0G
CSqGSIb3DQEBCwUAA4ICAQCUmJN/zwy2TPll+uw0yNv1x6t5RMNxJBPdvuqjY6ue
vXorcuIrayYQiaABsJLfeoQutuPBmCBYCKenLe1sKEFCTvUlhtihxAXBA/ABgi83
4HB+y/DeRcOAVNGMtjTTZkCt3JowM2o3DRt+H5ZAtHE1hWAjm9rAp4B+p0Q9auy0
olCThpgmU2vWBK8bWkqrpTOyWs4057J7usjZ97d/G3vJ8Wlk+XQ2MF/JUyqgIoNT
WQ4wGudoDMR4ggZIV/e6cIr2gZDQtFsOeKzw3hfUBymAL00m/cVgtu6C9sJxxcRj
cpLcC09I7Eh0ZTmILeMX5DxP+Y5csCfmD2zuny5CFdTy6EFcOm9MALh4JrmDX9Kc
Sz86SSxgcl/KbkJglyHE5c5kgu2QgxXkUemkt8TekbXS85GpRB/ReNjrztiYy+ST
q+flBebXWChFcqSWU2IyLCV+foT3XQvkYIGAjkU4XJuXS4KFVge5qh5s+c/ayxwU
wLeYRlg9DLQrIopC5rYV6K3tap4f0OzDh/dwWj2NubZRgL7rXteTAdXkzFnETbPy
+VQNWc9oXvbuJvsNj6hODsQthzTeZ15Nq8nYaLpO4umriWZ/orkekQsZq7u4daFU
CUTayCNk2VCbe2c45iVXgbUYLbf9m4wkUE00qFDjkZNbO/qd7lT65Wh6YMQJKJGC
Sw==
-----END CERTIFICATE-----""")

private_key = os.environ.get('Casdoor_PrivateKey', """-----BEGIN RSA PRIVATE KEY-----
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
-----END RSA PRIVATE KEY-----""")

sdk = CasdoorSDK(
    endpoint=os.environ.get('Casdoor_Endpoint', 'http://localhost:8000/'),
    client_id=os.environ.get('Casdoor_ClientId', 'a76a6289a3fa59f742b0'),
    client_secret=os.environ.get('Casdoor_ClientSecret', '5268177b8038ae2dc816c8c0e7d843d995063db7'),
    certificate=os.environ.get('Casdoor_Certificate', public_key),
    org_name=os.environ.get('Casdoor_OrgName', 'built-in'),
    application_name=os.environ.get('Casdoor_AppName', 'flashwords'),
)

# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

tinydb_instance = TinyDB("./review_history.json")  # TinyDB for review history


# Models
class Word(Base):
    __tablename__ = "words"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    foreign_word = Column(String, nullable=False)
    native_word = Column(String, nullable=False)
    description = Column(String)
    last_reviewed = Column(DateTime)


Base.metadata.create_all(bind=engine)


# Pydantic Schemas
class WordInput(BaseModel):
    foreignWord: str
    nativeWord: str
    description: Optional[str] = None


class WordResponse(WordInput):
    id: int
    lastReviewed: Optional[datetime]

    class Config:
        orm_mode = True


class ReviewResult(BaseModel):
    wordId: int
    rating: str
    reviewedAt: datetime


def getUserIDFromBearer(req: Request) -> str:
    token = req.headers["Authorization"]

    try:
        user_data = jwt.decode(token.split(" ")[1], public_key, algorithms=["RS256"], options={'verify_signature': False})
    except Exception as e:
        print("Error with ", token.split(" "))

    return user_data["id"]


# Constants
SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


class Settings(BaseModel):
    authjwt_algorithm: str = "RS256"
    # authjwt_decode_algorithms: str = "RS256"
    authjwt_public_key: str = public_key
    authjwt_private_key: str = private_key


@AuthJWT.load_config
def get_config():
    return Settings()


# FastAPI setup
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



@app.post("/users/{user_id}/words/import", status_code=200)
def import_words(req: Request, user_id: int = Path(...), words: List[WordInput] = Body(...), db=Depends(get_db)):
    user_id = getUserIDFromBearer(req)

    for word in words:
        db_word = Word(
            user_id=user_id,
            foreign_word=word.foreignWord,
            native_word=word.nativeWord,
            description=word.description,
        )
        db.add(db_word)
    db.commit()
    return {"message": "Words imported successfully"}


@app.get("/users/{user_id}/words")
def get_words(req: Request, user_id: int, db=Depends(get_db)):
    user_id = getUserIDFromBearer(req)

    words = db.query(Word).filter(Word.user_id == user_id).all()
    return words


@app.post("/users/{user_id}/words", status_code=201)
def add_word(req: Request, user_id: int, word: WordInput, db=Depends(get_db)):
    user_id = getUserIDFromBearer(req)

    db_word = Word(
        user_id=user_id,
        foreign_word=word.foreignWord,
        native_word=word.nativeWord,
        description=word.description,
    )
    db.add(db_word)
    db.commit()
    db.refresh(db_word)
    return db_word


@app.get("/users/{user_id}/words/{word_id}")
def get_word(req: Request, user_id: int, word_id: int, db=Depends(get_db)):
    user_id = getUserIDFromBearer(req)

    word = db.query(Word).filter(Word.user_id == user_id, Word.id == word_id).first()
    if not word:
        raise HTTPException(status_code=404, detail="Word not found")
    return word


@app.put("/users/{user_id}/words/{word_id}")
def update_word(req: Request, user_id: int, word_id: int, word: WordInput, db=Depends(get_db)):
    user_id = getUserIDFromBearer(req)

    db_word = db.query(Word).filter(Word.user_id == user_id, Word.id == word_id).first()
    if not db_word:
        raise HTTPException(status_code=404, detail="Word not found")
    db_word.foreign_word = word.foreignWord
    db_word.native_word = word.nativeWord
    db_word.description = word.description
    db.commit()
    db.refresh(db_word)
    return db_word


@app.delete("/users/{user_id}/words/{word_id}", status_code=204)
def delete_word(req: Request, user_id: int, word_id: int, db=Depends(get_db)):
    user_id = getUserIDFromBearer(req)

    db_word = db.query(Word).filter(Word.user_id == user_id, Word.id == word_id).first()
    if not db_word:
        raise HTTPException(status_code=404, detail="Word not found")
    db.delete(db_word)
    db.commit()

    # Remove from TinyDB review history
    tinydb_instance.remove(Query().word_id == word_id)
    return


from transformers import AutoModel, AutoTokenizer

# Initialize NLP components
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
transformer = AutoModel.from_pretrained("bert-base-uncased")
print('bert is loaded')





@app.get("/learning/next/{user_id}")
def get_next_word(req: Request, user_id: int, db=Depends(get_db)):
    user_id = getUserIDFromBearer(req)

    import torch
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from torch.nn import LSTM, Linear
    from torch.nn.functional import softmax

    class RNNScorer(torch.nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(RNNScorer, self).__init__()
            self.lstm = LSTM(input_size, hidden_size, batch_first=True)
            self.fc = Linear(hidden_size, output_size)

        def forward(self, x):
            _, (hn, _) = self.lstm(x)
            scores = self.fc(hn[-1])
            return softmax(scores, dim=-1)

    # Load words and history
    words = db.query(Word).filter(Word.user_id == user_id).all()
    if not words:
        raise HTTPException(status_code=204, detail="No words available for review")

    history = tinydb_instance.search(Query().user_id == user_id)
    current_time = datetime.now().timestamp()

    features = []
    labels = []
    word_map = {}
    word_embeddings = []

    # Extract features and embeddings
    for word in words:
        word_history = [h for h in history if h['word_id'] == word.id]
        total_reviews = len(word_history)
        avg_score = (
            np.mean([h['score'] for h in word_history])
            if word_history else 0
        )
        time_since_last_review = (
            current_time - max([h['timestamp'] for h in word_history], default=0)
            if word_history else float('inf')
        )
        is_reviewed = 0 if word.last_reviewed is None else 1

        # NLP embedding for the foreign word
        inputs = tokenizer(word.foreign_word, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            embedding = transformer(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy()

        word_embeddings.append(embedding)
        features.append([total_reviews, avg_score, time_since_last_review])
        labels.append(is_reviewed)
        word_map[len(features) - 1] = word

    # Replace infinite values
    features = np.array(features)
    features[np.isinf(features)] = 1e6

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # Combine tabular features and embeddings
    combined_features = np.hstack((X, word_embeddings))

    # Train Gradient Boosting model
    if len(set(labels)) > 1:
        boosting_model = GradientBoostingClassifier()
        boosting_model.fit(combined_features, labels)
        boosting_scores = boosting_model.predict_proba(combined_features)[:, 1]
    else:
        boosting_scores = [0] * len(features)

    # RNN-based ranking model
    rnn_input = torch.tensor(word_embeddings, dtype=torch.float32).unsqueeze(1)
    rnn_model = RNNScorer(input_size=rnn_input.shape[-1], hidden_size=128, output_size=2)
    rnn_scores = rnn_model(rnn_input).detach().numpy()[:, 1]

    # Final ranking by combining scores
    boosting_scores = np.array(boosting_scores)
    rnn_scores = np.array(rnn_scores)

    # Final ranking by combining scores
    final_scores = 0.7 * boosting_scores + 0.3 * rnn_scores
    ranked_indices = np.argsort(final_scores)[::-1]

    for idx in ranked_indices:
        word = word_map[idx]
        if word.last_reviewed is None:
            # word['foreign_word'] = 'kek'
            return randomQuestionFactory(word.id, word.native_word, word.foreign_word, word.description)

    for idx in ranked_indices:
        word = word_map[idx]
        if word.last_reviewed.timestamp() < current_time:
            try:
                print(word.foreign_word)
                return randomQuestionFactory(word.id, word.native_word, word.foreign_word, word.description)
            except Exception as e:
                print(e)
                return simpleQuestionFactory(word.id, word.native_word, word.foreign_word, word.description) 

    raise HTTPException(status_code=204, detail="No suitable words found for review")


@app.post("/learning/review/{user_id}", status_code=200)
def review_word(req: Request, user_id: int, review: ReviewResult, db=Depends(get_db)):
    user_id = getUserIDFromBearer(req)

    word = db.query(Word).filter(Word.user_id == user_id, Word.id == review.wordId).first()
    if not word:
        raise HTTPException(status_code=404, detail="Word not found")

    word.last_reviewed = review.reviewedAt
    db.commit()

    # Append to review history
    tinydb_instance.insert({
        'word_id': review.wordId,
        'user_id': user_id,
        'rating': review.rating,
        'score': map_rating_to_score(review.rating),
        'timestamp': review.reviewedAt.timestamp()
    })

    return {"message": "Review recorded successfully"}


# Helper functions for scoring
def map_rating_to_score(rating: str) -> int:
    rating_map = {
        "not_learned": 0,
        "almost_learned": 1,
        "learned": 2,
        "mastered": 3
    }
    return rating_map.get(rating, 0)


class WordStatsController:
    def __init__(self, db):
        self.db = db

    def get_word_reviews(self, user_id: int):
        return tinydb_instance.search(Query().user_id == user_id)

    def calculate_daily_statistics(self, reviews: List[Dict[str, Any]]):
        daily_stats = defaultdict(lambda: {"learned": 0, "almost_learned": 0, "not_learned": 0, "mastered": 0})

        for review in reviews:
            date = datetime.fromtimestamp(review["timestamp"]).date()
            rating = review["rating"]
            daily_stats[date][rating] += 1

        return daily_stats

    def calculate_word_difficulty(self, reviews: List[Dict[str, Any]]):
        word_scores = defaultdict(list)
        for review in reviews:
            word_scores[review["word_id"]].append(review["score"])

        word_difficulty = {
            word_id: statistics.mean(scores) for word_id, scores in word_scores.items()
        }
        return sorted(word_difficulty.items(), key=lambda x: x[1])

    def calculate_trends(self, reviews: List[Dict[str, Any]], period: int = 7):
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=period)

        trends = {"learned": 0, "almost_learned": 0, "not_learned": 0, "mastered": 0}

        for review in reviews:
            review_date = datetime.fromtimestamp(review["timestamp"]).date()
            if start_date <= review_date <= end_date:
                trends[review["rating"]] += 1

        return trends

    def calculate_forecasts(self, learned_count: int, target_word_count: int, days_left: int):
        daily_target = max(1, (target_word_count - learned_count) / days_left)
        return daily_target

    def calculate_user_progress(self, reviews: List[Dict[str, Any]]):
        progress = {"not_learned": 0, "almost_learned": 0, "learned": 0, "mastered": 0}
        for review in reviews:
            progress[review["rating"]] += 1
        return progress

    def calculate_gap_analysis(self, reviews: List[Dict[str, Any]], target_daily_reviews: int):
        daily_counts = defaultdict(int)

        for review in reviews:
            date = datetime.fromtimestamp(review["timestamp"]).date()
            daily_counts[date] += 1

        missed_days = {
            date: max(0, target_daily_reviews - count)
            for date, count in daily_counts.items()
        }

        return missed_days

    def calculate_individual_memory_coefficient(self, reviews: List[Dict[str, Any]]):
        total_reviews = len(reviews)
        successful_reviews = sum(1 for review in reviews if review["rating"] in ["learned", "mastered"])

        if total_reviews == 0:
            return 0.0

        return successful_reviews / total_reviews

    def calculate_user_metrics(self, user_id: int):
        reviews = self.get_word_reviews(user_id)

        daily_statistics = self.calculate_daily_statistics(reviews)
        difficulty_ranking = self.calculate_word_difficulty(reviews)
        weekly_trends = self.calculate_trends(reviews, period=7)
        monthly_trends = self.calculate_trends(reviews, period=30)

        learned_count = sum(1 for review in reviews if review["rating"] == "mastered")
        forecast = self.calculate_forecasts(learned_count, target_word_count=100, days_left=30)

        progress = self.calculate_user_progress(reviews)
        gap_analysis = self.calculate_gap_analysis(reviews, target_daily_reviews=10)
        memory_coefficient = self.calculate_individual_memory_coefficient(reviews)

        return {
            "daily_statistics": daily_statistics,
            "difficulty_ranking": difficulty_ranking,
            "weekly_trends": weekly_trends,
            "monthly_trends": monthly_trends,
            "forecast": forecast,
            "progress": progress,
            "gap_analysis": gap_analysis,
            "memory_coefficient": memory_coefficient,
            "total_words_reviewed": len(reviews),
            "learned_words_count": learned_count
        }


@app.get("/user/{user_id}/metrics")
def get_user_metrics(req: Request, user_id: int, db=Depends(get_db)):
    user_id = getUserIDFromBearer(req)

    controller = WordStatsController(db)
    metrics = controller.calculate_user_metrics(user_id)
    return metrics


# Weighted forgetting function (exponential decay)
def weighted_score(history, forgetting_rate=0.1):
    current_time = datetime.now().timestamp()
    scores = []
    for review in history:
        time_diff = current_time - review["timestamp"]
        weight = np.exp(-forgetting_rate * time_diff / (60 * 60 * 24))
        scores.append(review["score"] * weight)
    return sum(scores) / sum(
        [np.exp(-forgetting_rate * (current_time - review["timestamp"]) / (60 * 60 * 24)) for review in
         history]) if scores else 0


@app.get("/stats/progress/{user_id}")
def get_progress(req: Request, user_id: str, db=Depends(get_db)):
    user_id = getUserIDFromBearer(req)

    words = db.query(Word).filter(Word.user_id == user_id).all()
    history = tinydb_instance.search(Query().user_id == user_id)

    progress_data = {}
    for word in words:
        word_history = [h for h in history if h["word_id"] == word.id]
        progress_series = []
        for review in sorted(word_history, key=lambda h: h["timestamp"]):
            progress_series.append({
                "time": datetime.fromtimestamp(review["timestamp"]).isoformat(),
                "score": weighted_score(word_history[:word_history.index(review) + 1])
            })
        progress_data[word.foreign_word] = progress_series

    return progress_data
