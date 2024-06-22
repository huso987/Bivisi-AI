import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn


df = pd.read_csv('categories.csv')
df['combined'] = df['category'] + " " + df['subcategory']

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined'])

class Query(BaseModel):
    query: str

def get_recommendations(query, tfidf_matrix, df, top_n=5):
    query_vec = tfidf.transform([query])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-top_n-1:-1]
    
    recommendations = []
    for idx in related_docs_indices:
        subcategory = df.iloc[idx]['subcategory']
        category = df.iloc[idx]['category']
        recommendations.append({'subcategory': subcategory, 'category': category})

    return recommendations

app = FastAPI()

@app.post("/recommendations/")
def get_recommendation(query: Query):
    recommendations = get_recommendations(query.query, tfidf_matrix, df)
    return {"recommendations": recommendations}

# To run the server, use the following command:
# uvicorn app:app --reload

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

# You can see the automatically generated documentation 
# of the API by going to http://127.0.0.1:8000/docs from your browser.