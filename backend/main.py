import statistics
from collections import defaultdict

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

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
import bitsandbytes

from llama_cpp import Llama

# llm = Llama.from_pretrained(
# 	repo_id="bartowski/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF",
# 	filename="Llama-3.1-8B-Lexi-Uncensored-V2-Q8_0.gguf",
# )
#
#
#
#
# input_text = 'Make up a sentence with the missing word combination “sun” and write a triplet in its place. Return the answer in json format, example {“sentence”: “She said ... Bob"}'
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



public_key = """-----BEGIN CERTIFICATE-----
MIIE3TCCAsWgAwIBAgIDAeJAMA0GCSqGSIb3DQEBCwUAMCgxDjAMBgNVBAoTBWFk
bWluMRYwFAYDVQQDEw1jZXJ0LWJ1aWx0LWluMB4XDTI1MDEwODE0MzAzMloXDTQ1
MDEwODE0MzAzMlowKDEOMAwGA1UEChMFYWRtaW4xFjAUBgNVBAMTDWNlcnQtYnVp
bHQtaW4wggIiMA0GCSqGSIb3DQEBAQUAA4ICDwAwggIKAoICAQDEdHDFbn2yE+Pb
qbzY2PRfYuhgo0cqpCArtEPCbyyz6sD3b9+6R2D/4iZSnGlGgEdZS5mEPZGlmOLo
A7dRrw4fT9tx02zCPON4ubsllwU/BhWZqy7q+Q1dV8RrrA4xSgb2g1F9R2R8i784
ll11p24VwUX4zAnhk+5QAe8Wcu6K6VbF+DmndoNMBB7CklcF+jJ+GBxXKrSyo3s3
9UCLGEP5bBuDwsKCknvMwFmC23cDpYt+YQnxzxgGsws64WNlEBx8UxWNTxkBu6jW
/5IlvV+19BmowsAzvybGGqaD5leBzqEESW9pLvuRREuch90GDCk9pJdTsxilurwF
fJLWV1KNjURZTxAHVLek2FMzn/A9NL20X7acnLpju5eqSDlwSX097auLZqRP+i3s
7S63z8/k1FNPlze3MNdgDb+0zQaX7XpjvUO2hhedUfZkFyQzUSxYxNKBaq6RTOhv
6sGs0fUQMYNCjfIrehWtl6djZ8sskl86fXOGZ/m4HekXxwSm5tFMT/DTueWTO6x2
pTjfd24D4asOUTjy8dR0k4SxcN1FuTAXUeRwDH+MmvQcomI7v7tmng1Sl+Wfn01l
iVP0VNZscOI/JNNth9gKSfystyRjbL0G9QJ8GsOLBburlOnzOG/2nOoTR7SCoHaT
e3+bgSSemxNC/lSgGY4ZHpSCXZugUQIDAQABoxAwDjAMBgNVHRMBAf8EAjAAMA0G
CSqGSIb3DQEBCwUAA4ICAQCCOoUWJHWXnfuWvkhy5LWTDa8nS3mHeKsuoY7cNahc
utcsS8r2QUexeKQ/ddudR8634HPznoXe+sLTRSgYd5fSIuvl73ekmhcmOda7r7bH
E+YjJsBSz6hGIf5+h6SWClIgY+U+SZlEMd+xeS31UJLf5ju1MGYHBQ5N306JNvNL
OEvjFxVI+1s5gyhDbkW3eRYCSSq3XjdrAre0pAAXs409a9KOacgkfzU0cjAkMV2I
kc/HEzlj416usYC++jVKX1yDbREngzJnGTnaPTCQodlhsjaVUduDIzGORto+jGqQ
EzisDgQEr+taRkRBBbs5Etg7BWqNaftmEB89ShuFTW2s7o4dkZhHs8ejjDmt+G0T
QSA5rzPILe0A6YdIxQSFjZyMLbiTL9nTRJl9qigokP07BSFDjzAjrZ8cjUs8tQrd
9XyhGcezjR6OgCQEIH5YmeyC4UA123egevWcicjBRH4i6fMgEbS8biZuPReP8wr4
nJb0WqOpCaPn9UuIxFMSDfiRAk5+190R0eAuTPnKAxKUMsqq5T5pNKdQhY7uTSCB
PH+iIrlG+Im1jNcCd6maG9IGH3R5mjd65DwtZp1MUaTDg97bNIk/4KhGxYLJu7pi
AzasmLZOrT7KrNqEBPQoznBazpINgRGVVZG5T2juXLA0z6NiCSEC8bkKla81kBgK
eQ==
-----END CERTIFICATE-----"""

private_key = """-----BEGIN RSA PRIVATE KEY-----
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
+YZvAoIBAHaSD1Q2grLZZePD7SKp/vlX/OaQFNmWekad50t5oq6UBIK1KghiAeCe
PT7eENP7HSkdZpfvGlnerHYb1p4eT88Vbt/p2GUndvZeekielgxG2y1DFd8KkjRk
wXznkEBwtvTbFJ5rC0GHyAsIlr1YhOBIQ/zF7LLtbOmkiJPGB88emCVtfyq37M55
hVBuBhrTTNU7Rvaa8LYOzffY6090jK+5TslgeCEJ/F7qm+JkNZBIKhXY+69mfJXh
d0QHldnCZE9Gn7+bSp03qGi+zFmOnxeDagHVAZ8/+Hum0o/vsZovKVaWu/8xCfe6
1jWxzBxfpyCfJ1LHhwehjKTlysLkijsCggEBAOrNm3OThds410kNhGm2gvlEgjWn
7mnGUFFIcgibrd/E74k+X+abp2nkDCOyP0mBBH3NlobD2dEyLN74Xp1HM4dU1Otc
rAlPYlah8p5oFUm3127tSO3ITzeyoNj5tEZL1Tcwy/DUuALTTXP7Uf/szw4FyFp2
E2eP3zYeKUGv1L1AXkrA7OV759aHjrUIXU5Mb7W9XP3YzOtxqvAKTpc9W3s4cMQ3
5ngHJJx1pz62tcLcr5sOF0btqB+Wh1Z2bKMq6RbSRlLc+zcJoh8UdUJ9jATIMg5G
JvcGvYcdxQjyTkrldabE/0JcnqYG+bZCDcwrrdGdR9Dj6bEjGl/Xh2NWOTs=
-----END RSA PRIVATE KEY-----"""

sdk = CasdoorSDK(
    endpoint='http://localhost:8000/',
    client_id='26bb0c442dd7c4bb8dbc',
    client_secret='770cb3030e11c1a22aa27cf24244c9b060315b81',
    certificate=public_key,
    org_name='built-in',
    application_name='application_vocabularius',
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
    user_data = jwt.decode(token.split(" ")[1], public_key, algorithms=["RS256"], options={'verify_signature': False})
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


# Routes
@app.post("/auth")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    token = sdk.get_oauth_token("c72f94308551cbee4d6a")
    user = sdk.parse_jwt_token(token["access_token"])

    return {"accessToken": token["access_token"], "tokenType": "bearer"}


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
            # print(json.dumps(word))
            # word['foreign_word'] = 'kek'
            return word

    for idx in ranked_indices:
        word = word_map[idx]
        if word.last_reviewed.timestamp() < current_time:
            # print(json.dumps(word))
            # word['foreign_word'] = 'kek'
            return word

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
def get_progress(user_id: str, db=Depends(get_db)):
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
