import pytest
from unittest.mock import patch
from app import questions
from app import llm
import requests_mock


@pytest.fixture
def mock_llm_server():
    response_json = {
        "model": "Hudson/llama3.1-uncensored:8b",
        "created_at": "2025-01-13T22:04:13.3769876Z",
        "response": "{\"response\": \"Привет...\"}",
        "done": True,
        "done_reason": "stop",
        "context": [128006, 882, 128007, 271, 19871, 59030, 19039, 116049, 17618, 5524, 104171, 43293, 111106,
                    92207, 12507, 108349, 8341, 7740, 105019, 104171, 107541, 91105, 1840, 30480, 113719, 8341,
                    38438, 50306, 13, 107447, 23446, 93747, 5927, 54261, 123763, 3024, 5524, 5524, 57719, 59921,
                    28241, 6578, 116049, 102345, 5927, 364, 2376, 4527, 39903, 19039, 38098, 4898, 2233, 30656,
                    34613, 93747, 92457, 44216, 93608, 101706, 33742, 11, 109389, 119047, 7740, 114455, 47295,
                    121869, 101079, 13, 119801, 54261, 8131, 59921, 39280, 7740, 115119, 23630, 1506, 45022, 720,
                    128009, 128006, 78191, 128007, 271, 5018, 2376, 794, 330, 54745, 28089, 8341, 1131, 9388],
        "total_duration": 2337482700,
        "load_duration": 16473000,
        "prompt_eval_count": 87,
        "prompt_eval_duration": 436000000,
        "eval_count": 10,
        "eval_duration": 1884000000
    }
    with requests_mock.Mocker() as m:
        m.post("http://localhost:11434/api/generate", json=response_json)
        yield


@pytest.fixture
def mock_get_sentence(mock_llm_server):
    with patch('app.llm.get_sentence') as mock:
        mock.return_value = "The ... is very beautiful."
        yield mock

@pytest.fixture
def mock_get_ru_native_sentence(mock_llm_server):
    with patch('app.llm.get_ru_native_sentence') as mock:
        mock.return_value = "Он увидел ... на столе."
        yield mock

def test_simple_question_factory():
    question = questions.simpleQuestionFactory("привет", "hello", "greeting")
    assert question["type"] == "simpleQuestion"
    assert question["question"]["value"] == "hello"
    assert question["answer"]["value"] == "привет"
    assert question["answer"]["description"] == "greeting"

def test_inverse_question_factory():
    question = questions.inverseQuestionFactory("привет", "hello", "greeting")
    assert question["type"] == "inverseQuestion"
    assert question["question"]["value"] == "привет"
    assert question["answer"]["value"] == "hello"
    assert question["answer"]["description"] == "greeting"

def test_sentence_simple_question_factory(mock_get_sentence):
    question = questions.sentenceSimpleQuestionFactory("привет", "hello", "greeting")
    mock_get_sentence.assert_called_once_with("hello")
    assert question["type"] == "sentenceSimpleQuestion"
    assert question["question"]["value"] == "The ... is very beautiful."
    assert question["answer"]["value"] == "hello (привет)"
    assert question["answer"]["description"] == "greeting"

def test_inverse_sentence_simple_question_factory(mock_get_ru_native_sentence):
    question = questions.inverseSentenceSimpleQuestionFactory("привет", "hello", "greeting")
    mock_get_ru_native_sentence.assert_called_once_with("привет")
    assert question["type"] == "inverseSentenceSimpleQuestion"
    assert question["question"]["value"] == "Он увидел ... на столе."
    assert question["answer"]["value"] == "hello (привет)"
    assert question["answer"]["description"] == "greeting"

def test_random_question_factory(mock_get_sentence, mock_get_ru_native_sentence):
    with patch('random.choice') as mock_choice:
        mock_choice.side_effect = [
            questions.simpleQuestionFactory,
            questions.inverseQuestionFactory,
            questions.sentenceSimpleQuestionFactory,
            questions.inverseSentenceSimpleQuestionFactory,
        ]
        
        question1 = questions.randomQuestionFactory("привет", "hello", "greeting")
        assert question1["type"] == "simpleQuestion"

        question2 = questions.randomQuestionFactory("привет", "hello", "greeting")
        assert question2["type"] == "inverseQuestion"

        question3 = questions.randomQuestionFactory("привет", "hello", "greeting")
        assert question3["type"] == "sentenceSimpleQuestion"

        question4 = questions.randomQuestionFactory("привет", "hello", "greeting")
        assert question4["type"] == "inverseSentenceSimpleQuestion"
