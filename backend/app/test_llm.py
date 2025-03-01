import pytest
from unittest.mock import patch
from llm import questions
from llm import llm

question = questions.sentenceSimpleQuestionFactory("привет", "hello", "greeting")