import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ratings_dict = {
    'Titanic': [5, 5, 1, 2, 1],
    'Avatar': [4, 5, 1, 2, 1],
    'Avengers': [1, 2, 5, 5, 4],
    'Iron Man': [1, 1, 5, 4, 5],
    'Interstellar': [5, 4, 2, 1, 1],
    'The Notebook': [5, 5, 1, 1, 1],
    'Batman Begins': [1, 2, 5, 4, 5],
    'Spider-Man': [1, 1, 5, 5, 4],
    'Thor': [1, 2, 5, 4, 4],
    'Black Panther': [1, 2, 5, 5, 5]
}

ratings = pd.DataFrame(ratings_dict)

similarity = cosine_similarity(ratings.T)

similarity_df = pd.DataFrame(
    similarity,
    index=ratings.columns,
    columns=ratings.columns
)

print("Available Movies:\n")

for movie in similarity_df.columns:
    print(movie)

movie_name = input("\nEnter Movie Name: ")

movie_name = movie_name.title()

if movie_name not in similarity_df.columns:
    print("\nMovie not found")

else:
    recommendations = similarity_df[movie_name].sort_values(ascending=False)

    print("\nRecommended Movies:\n")

    for movie in recommendations.index[1:6]:
        print(movie)

print("\nConclusion:")
print("The Recommendation System successfully recommends movies using Collaborative Filtering and Cosine Similarity techniques.")
