import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the CSV file
Films = pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\vs code\recommendation system\movies.csv")

# Fill missing values with an empty string
Films = Films.fillna('')

# Check if 'genres' and 'title' columns exist and are spelled correctly
if 'genres' in Films.columns and 'title' in Films.columns:
    
    # Combine relevant features (in this case, just 'genres')
    Films['features'] = Films['genres']

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(Films['features'])

    cosine_sim = cosine_similarity(tfidf_matrix, dense_output=False)

    def get_recommendations(title, cosine_sim=cosine_sim):
        if title in Films['title'].values:
            idx = Films[Films['title'] == title].index[0]
            sim_scores = list(enumerate(cosine_sim[idx].toarray()[0]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:4]  # Get the top 3 similar movies
            movie_indices = [i[0] for i in sim_scores]
            return Films['title'].iloc[movie_indices]
        else:
            return ["Movie not found in the database."]

    user_preference = 'Comedy'
    recommended_movies = Films[Films['genres'].str.contains(user_preference, case=False)]['title'].tolist()
    print(f"Recommended movies in {user_preference} genre:")
    for movie in recommended_movies:
        print(f"- {movie}")

    movie_title = 'Shayar'
    print(f"\nMovies similar to '{movie_title}':")
    similar_movies = get_recommendations(movie_title)
    for movie in similar_movies:
        print(f"- {movie}")
else:
    print("One or more of the expected columns are missing.")


