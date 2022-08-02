import pandas as pd
import numpy as np
import warnings;

warnings.filterwarnings('ignore')

movies = pd.read_csv('C:/Users/ForYou/Desktop/ml-25m/movies.csv')
ratings = pd.read_csv('C:/Users/ForYou/Desktop/ml-25m/ratings.csv')

print(movies.shape, ratings.shape)

print(movies.head(2))
print(ratings.head(2))

ratings = ratings.truncate(before=0, after=1000000, axis=0)

print(movies.shape, ratings.shape)

rating_movies = pd.merge(ratings, movies, on='movieId')
print(rating_movies.head(2))

ratings_matrix = rating_movies.pivot_table('rating', index='userId', columns='title')
ratings_matrix.fillna(0, inplace=True)
print(ratings_matrix.head(2))

ratings_matrix_T = ratings_matrix.transpose()

print(ratings_matrix_T.head(2))

from sklearn.metrics.pairwise import cosine_similarity

item_sim = cosine_similarity(ratings_matrix_T, ratings_matrix_T)
item_sim_df = pd.DataFrame(data=item_sim, index=ratings_matrix.columns, columns=ratings_matrix.columns)

print(item_sim_df.shape)
print(item_sim_df.head(2))


def find_sim_movie_item(df, title_name, top_n=10):
    title_movie_sim = df[[title_name]].drop(title_name, axis=0)

    return title_movie_sim.sort_values(title_name, ascending=False)[:top_n]


print(find_sim_movie_item(item_sim_df, 'Godfather, The (1972)'))
