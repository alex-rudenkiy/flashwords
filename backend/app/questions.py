import random

from llm import get_ru_native_sentence, get_sentence


def randomQuestionFactory(word_id, nativeWord, foreignWord, description):
    factories = [simpleQuestionFactory, inverseQuestionFactory, sentenceSimpleQuestionFactory]
    factory = random.choice(factories)
    result = factory(word_id, nativeWord, foreignWord, description)
    return result

def simpleQuestionFactory(word_id, nativeWord, foreignWord, description):
    question = {
        "type": "simpleQuestion",
        "id": word_id,
        "question": {
            "value": foreignWord
        },
        "answer": {
            "value": nativeWord,
            "description": description
        }
    }
    return question

def inverseQuestionFactory(word_id, nativeWord, foreignWord, description):
    question = {
        "type": "inverseQuestion",
        "id": word_id,
        "question": {
            "value": nativeWord
        },
        "answer": {
            "value": foreignWord,
            "description": description
        }
    }
    return question

def sentenceSimpleQuestionFactory(word_id, nativeWord, foreignWord, description):
    question = {
        "type": "sentenceSimpleQuestion",
        "id": word_id,
        "question": {
            "value": get_sentence(foreignWord)
        },
        "answer": {
            "value": f"{foreignWord} ({nativeWord})",
            "description": description
        }
    }
    return question

def inverseSentenceSimpleQuestionFactory(word_id, nativeWord, foreignWord, description):
    question = {
        "type": "inverseSentenceSimpleQuestion",
        "id": word_id,
        "question": {
            "value": get_ru_native_sentence(nativeWord)
        },
        "answer": {
            "value": f"{foreignWord} ({nativeWord})",
            "description": description
        }
    }
    return question
