import json
import os

import requests
import re
import json
regexPattern = r"({\"response\":.*})"

llm_endpoint = os.environ.get('LLM_Endpoint', 'http://localhost:11434')

def get_sentence(word):
    prompt = {
        "model": "Hudson/llama3.1-uncensored:8b",
        "stream": False,
        "prompt": "Make a sentence with the missing word "+word+" and enter an ellipsis in its place. Return the response in json format and the composed sentence in the \"response\" field. Give a specific answer without comments and other garbage. Without formatting and \n. Example given the word \"car\", it will be correct to return \"{\"response\": \"I drove my new ... to the village.\"}"
    }
    
    response = requests.post(f"{llm_endpoint}/api/generate", json=prompt)
    print(response.json())

    return json.loads(re.findall(regexPattern,json.loads(response.text)['response'])[0])['response']

def get_ru_native_sentence(word):
    prompt = {
        "model": "Hudson/llama3.1-uncensored:8b",
        "stream": False,
        "prompt": "Сделай предложение с пропущенным словом "+word+" и где пропуск напиши троеточие. Верни ответ в формате json с сформированным предложением в 'response'. Дай конкректный ответ без комментариев, пояснений и прочего мусора. Без форматирования и переноса строк \n"
    }

    response = requests.post(f"{llm_endpoint}/api/generate", json=prompt)
    print(response.json())

    return json.loads(re.findall(regexPattern,json.loads(response.text)['response'])[0])['response']
