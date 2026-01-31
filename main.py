import uvicorn
from fastapi import FastAPI, Body
import numpy as np
import pandas as pd
import pickle
import nltk
from nltk.tokenize import word_tokenize
from fastapi import HTTPException

from nltk.corpus import stopwords


app = FastAPI()
pickle_in = open("model.pkl", "rb")
model = pickle.load(pickle_in)

pickle_in_2 = open("vectorizer.pkl", "rb")
vector = pickle.load(pickle_in_2)

pickle_in_3 = open("lemmatizer.pkl", "rb")
lemmatizer = pickle.load(pickle_in_3)

@app.post('/predict')
def spam_ham(text: str = Body(...,
        media_type="text/plain",
        title="Enter Text",
        description="Paste your email or message here"
    )
):
    # data = data.model_dump()  
    # print(data)
    # text = data['Enter_mail_in_one_para']
    text = text.lower()
    word = word_tokenize(text)  
    for i in range(0,len(word)):
        if word[i] not in stopwords.words("english"):
            word[i] = lemmatizer.lemmatize(word[i])
            # word[i] = new_word
        else:
            word[i] = " "
    text = " ".join(word)
    text = vector.transform([text]).toarray()

    prediction = model.predict(text)

    if prediction[0] == 0:
        return {"prediction": "It's a ham mail."}
    else:
        return {"prediction": "It's a spam mail."}
    
if __name__ == '__main__':
    uvicorn.run(app, host = '127.0.0.1', port = 8000)