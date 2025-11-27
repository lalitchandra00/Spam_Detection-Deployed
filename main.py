import uvicorn
from fastapi import FastAPI
from values import Values
import numpy as np
import pandas as pd
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from fastapi import HTTPException

from nltk.corpus import stopwords


app = FastAPI()
pickle_in = open("model.pkl", "rb")
model = pickle.load(pickle_in)

pickle_in_2 = open("vectorizer.pkl", "rb")
vector = pickle.load(pickle_in_2)


# @app.post('/predict')
# def spam_ham(data: Values):
#     data = data(dict)
#     print(data)
#     text = data[text]
#     text = text.lower()
#     word = word_tokenize(text)
#     for i in range(0,len(word)):
#         if word[i] not in stopwords.words("english"):
#             new_word = lemmatizer(WordNetLemmatizer(word[i]))
#             word[i] = new_word
#         else:
#             word[i] = " "
#     text = vector.fit_transform(text).toarray()

#     prediction = model.predict(text)
#     return prediction




@app.post('/predict')
def spam_ham(data: Values):
    try:
        # Type checking for input data
        if not hasattr(data, 'text') or not isinstance(data.text, str):
            raise HTTPException(status_code=400, detail="Input must have a 'text' field of type string.")

        text = data.text
        text = text.lower()
        words = word_tokenize(text)
        processed_words = []
        for word in words:
            if word not in stopwords.words("english"):
                new_word = lemmatizer.lemmatize(word)
                processed_words.append(new_word)
        processed_text = " ".join(processed_words)
        text_vector = vector.fit_transform([processed_text]).toarray()

        prediction = model.predict(text_vector)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

    
if __name__ == '__main__':
    uvicorn.run(app, host = '127.0.0.1', port = 8000)