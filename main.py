from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import random

app = FastAPI()

# Sample Data
selected_columns = ['Title', 'Tag', 'Review', 'Comment', 'Address', 'Country', 'Price', 'Rating', 'tags', 'Governorate']

df = pd.read_csv("https://raw.githubusercontent.com/ahmedmostafa147/BackEnd/main/final_data.csv")[selected_columns].dropna()

df['Tag'] = df['Tag'].astype(str)
df['Review'] = df['Review'].astype(str)
df['Comment'] = df['Comment'].astype(str)

# Feature Engineering
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Tag'] + ' ' + df['Review'] + ' ' + df['Comment'])

class RecommendationRequest(BaseModel):
    country: str
    governorate: str
    survey_responses: list
    num_days: int
    budget: float

class Recommendation(BaseModel):
    Title: str
    Price: float
    Tags: str
    Governorate: str
    Day: int

@app.post("/recommendations/", response_model=list[Recommendation])
async def get_recommendations(recommendation_request: RecommendationRequest):
    country = recommendation_request.country
    governorate = recommendation_request.governorate
    survey_responses = recommendation_request.survey_responses
    num_days = recommendation_request.num_days
    budget = recommendation_request.budget
    
    filtered_df = df[(df['Country'] == country) & (df['Governorate'] == governorate)]

    if filtered_df.empty:
        raise HTTPException(status_code=404, detail="No data found for the specified country and governorate.")
    
    user_profile = f"{country} {governorate} {' '.join(survey_responses)}"
    user_profile_vectorized = tfidf_vectorizer.transform([user_profile])
    places_vectorized = tfidf_vectorizer.transform(filtered_df['tags'])
    sim_scores = linear_kernel(user_profile_vectorized, places_vectorized).flatten()

    if not any(response_indices for response in survey_responses for response_indices in [i for i, tag in enumerate(filtered_df['tags']) if response.lower() in tag.lower()]):
        raise HTTPException(status_code=404, detail="No suitable places found for the given survey responses.")
    
    max_price_per_day = budget / num_days

    recommendations_df = pd.DataFrame(columns=['Title', 'Price', 'tags', 'Governorate', 'Day'])

    for day in range(1, num_days + 1):
        daily_recommendations = []

        if day == 1:
            hotel_recommendation = filtered_df[filtered_df['tags'].str.lower().str.contains('hotel') & (filtered_df['Price'] <= max_price_per_day)].sample(1)[['Title', 'Price', 'tags', 'Governorate']]
            daily_recommendations.append(hotel_recommendation)

        restaurant_recommendation = filtered_df[filtered_df['tags'].str.lower().str.contains('restaurant') & (filtered_df['Price'] <= max_price_per_day)].sample(1)[['Title', 'Price', 'tags', 'Governorate']]
        daily_recommendations.append(restaurant_recommendation)

        for response in survey_responses:
            response_indices = [i for i, tag in enumerate(filtered_df['tags']) if response.lower() in tag.lower()]

            if response_indices:
                valid_indices = [idx for idx in response_indices if filtered_df.iloc[idx]['Price'] <= max_price_per_day]
                if valid_indices:
                    random_index = random.choice(valid_indices)
                    recommendation = filtered_df.iloc[[random_index]][['Title', 'Price', 'tags', 'Governorate']]
                    daily_recommendations.append(recommendation)

        for recommendation in daily_recommendations:
            recommendation['Day'] = day
            recommendations_df = pd.concat([recommendations_df, recommendation])

    recommendations = []
    for i, recommendation in recommendations_df.iterrows():
        recommendations.append({
            "Title": recommendation['Title'],
            "Price": recommendation['Price'],
            "Tags": recommendation['tags'],
            "Governorate": recommendation['Governorate'],
            "Day": recommendation['Day']
        })
    
    return recommendations
# Path: main.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
