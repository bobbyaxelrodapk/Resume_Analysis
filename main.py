# Create an application
from fastapi import FastAPI
from fastapi import File, UploadFile
import pandas as pd
import numpy as np
import joblib
import fitz
import pickle
import sys
from fastapi.staticfiles import StaticFiles
import matplotlib.pyplot as plt
import uuid
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request

# Importing our module
from person import Model
from keywordextract import extract_key_word
from charts import create_charts

app = FastAPI()

traits = ['OPN', 'CON', 'EXT', 'AGR', 'NEU']
labels=['0-2.5 LPA','2.5-5 LPA','5-7.5 LPA','7.5-10 LPA','10-12.5 LPA','12.5-15 LPA','12.5-15 LPA','15-17.5 LPA','17.5-20 LPA','25-30  LPA','20-30 LPA','30-40 LPA','40 LPA -2 CrPA']

setattr(sys.modules['__main__'], 'Model', Model)

models = {}
for trait in traits:
    with open('static/model/' + trait + '_model.pkl', 'rb') as f:
        models[trait] = pickle.load(f)
clf = joblib.load('static/model/salaryclassifier.pickle')
tfidf_vectorizer = pickle.load(open('static/model/fitted_vectorizer.pickle','rb'))
        
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="static/templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/upload")
def upload(request: Request, file: UploadFile = File(...)):
    uid = str(uuid.uuid4())
    u_filename = f"{uid}_{file.filename}"
    
    ## File to text object
    try:
        contents = file.file.read()
        with open(f"static/resume/{u_filename}", 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()
    doc1=fitz.open(f"static/resume/{u_filename}")
    output1 = ""
    for page in doc1:
        text=page.get_text()
        output1 =" ".join([output1, text])
    text=" ".join(output1.split())

    ## Personality prediction
    predictions = {'score':[], 'prob': []}
    for trait in traits:
        pkl_model = models[trait]
        trait_scores = pkl_model.predict([text], regression=True).reshape(1, -1)
        predictions['score'].append(trait_scores.flatten()[0])
        trait_categories_probs = pkl_model.predict_proba([text])
        predictions['prob'].append(trait_categories_probs[:, 1][0])
 
    ## Salary prediction
    result = clf.predict(tfidf_vectorizer.transform([text]))
    
    ## Key word extraction
    predictions['key_word']= extract_key_word(text)
    print(predictions['key_word'])
    create_charts(predictions, text, uid)

    
    return templates.TemplateResponse("display.html", {
        "request": request, 
        "word_cloud": f"/static/image/{uid}_wordcloud.png", 
        "salary": labels[result[0]], 
        "key_word": predictions["key_word"], 
        "trait_prob": f"/static/image/{uid}_traitprob.png", 
        "trait_score": f"/static/image/{uid}_traitscore.png",
        "key_word_count": f"/static/image/{uid}_keyword.png"
    })

