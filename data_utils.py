"""
Data utilities for loading and processing movie data
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_sample_data():
    """
    Generate sample movie and rating data for demonstration
    
    Returns:
        tuple: (movie_data, ratings_data) DataFrames
    """
    # Sample movie data
    movies = {
        'movieId': range(1, 51),
        'title': [
            'The Dark Knight', 'Inception', 'Pulp Fiction', 'The Matrix', 'Fight Club',
            'Forrest Gump', 'The Shawshank Redemption', 'Interstellar', 'The Godfather',
            'Goodfellas', 'The Avengers', 'The Lord of the Rings', 'Gladiator',
            'Saving Private Ryan', 'The Lion King', 'Titanic', 'Jurassic Park',
            'Star Wars: Episode IV', 'Blade Runner', 'The Terminator',
            'Alien', 'Aliens', 'Predator', 'RoboCop', 'Total Recall',
            'The Fifth Element', 'Starship Troopers', 'Independence Day',
            'Men in Black', 'The Matrix Reloaded', 'The Matrix Revolutions',
            'Iron Man', 'Spider-Man', 'Batman Begins', 'The Dark Knight Rises',
            'Superman', 'Wonder Woman', 'Black Panther', 'Thor', 'Captain America',
            'X-Men', 'X-Men: Days of Future Past', 'Deadpool', 'Logan',
            'The Incredibles', 'Toy Story', 'Monsters Inc', 'Finding Nemo',
            'Up', 'Wall-E'
        ],
        'genres': [
            'Action|Crime|Drama', 'Action|Sci-Fi|Thriller', 'Crime|Drama',
            'Action|Sci-Fi', 'Drama|Thriller', 'Drama|Romance',
            'Drama', 'Adventure|Drama|Sci-Fi', 'Crime|Drama',
            'Biography|Crime|Drama', 'Action|Adventure|Sci-Fi',
            'Adventure|Drama|Fantasy', 'Action|Adventure|Drama',
            'Action|Drama|War', 'Animation|Adventure|Drama',
            'Drama|Romance', 'Action|Adventure|Sci-Fi',
            'Action|Adventure|Fantasy', 'Action|Sci-Fi|Thriller',
            'Action|Sci-Fi', 'Horror|Sci-Fi|Thriller',
            'Action|Adventure|Sci-Fi', 'Action|Adventure|Horror',
            'Action|Crime|Sci-Fi', 'Action|Adventure|Sci-Fi',
            'Action|Adventure|Comedy|Sci-Fi', 'Action|Adventure|Sci-Fi',
            'Action|Adventure|Sci-Fi', 'Action|Adventure|Comedy|Sci-Fi',
            'Action|Sci-Fi', 'Action|Sci-Fi', 'Action|Adventure|Sci-Fi',
            'Action|Adventure|Sci-Fi', 'Action|Crime|Drama',
            'Action|Crime|Drama', 'Action|Adventure|Sci-Fi',
            'Action|Adventure|Fantasy', 'Action|Adventure|Sci-Fi',
            'Action|Adventure|Fantasy', 'Action|Adventure|Sci-Fi',
            'Action|Adventure|Sci-Fi', 'Action|Adventure|Sci-Fi',
            'Action|Adventure|Comedy', 'Action|Drama|Sci-Fi',
            'Animation|Action|Adventure|Comedy|Family',
            'Animation|Adventure|Comedy|Family',
            'Animation|Adventure|Comedy|Family',
            'Animation|Adventure|Comedy|Drama|Family',
            'Animation|Adventure|Comedy|Drama|Family',
            'Animation|Adventure|Comedy|Family'
        ]
    }
    
    movie_data = pd.DataFrame(movies)
    
    # Generate sample ratings data
    np.random.seed(42)
    n_users = 100
    n_movies = 50
    ratings = []
    
    # Create ratings with some patterns (users tend to rate similar movies similarly)
    for user_id in range(1, n_users + 1):
        # Each user rates 10-30 movies
        n_ratings = np.random.randint(10, 31)
        
        # Users have genre preferences
        preferred_genres = np.random.choice(
            ['Action', 'Drama', 'Sci-Fi', 'Comedy', 'Horror'],
            size=2,
            replace=False
        )
        
        rated_movies = np.random.choice(range(1, n_movies + 1), size=n_ratings, replace=False)
        
        for movie_id in rated_movies:
            movie_genres = movie_data[movie_data['movieId'] == movie_id]['genres'].values[0]
            
            # Higher ratings for preferred genres
            if any(genre in movie_genres for genre in preferred_genres):
                rating = np.random.normal(4.0, 0.8)
            else:
                rating = np.random.normal(3.0, 1.0)
            
            # Clamp rating between 1 and 5
            rating = max(1.0, min(5.0, round(rating, 1)))
            
            ratings.append({
                'userId': user_id,
                'movieId': movie_id,
                'rating': rating
            })
    
    ratings_data = pd.DataFrame(ratings)
    
    return movie_data, ratings_data


def load_data_from_files(movies_file, ratings_file):
    """
    Load movie and ratings data from CSV files
    
    Args:
        movies_file: Path to movies CSV file
        ratings_file: Path to ratings CSV file
        
    Returns:
        tuple: (movie_data, ratings_data) DataFrames
    """
    movie_data = pd.read_csv(movies_file)
    ratings_data = pd.read_csv(ratings_file)
    
    return movie_data, ratings_data


def save_data_to_csv(movie_data, ratings_data, output_dir='data'):
    """
    Save movie and ratings data to CSV files
    
    Args:
        movie_data: DataFrame with movie information
        ratings_data: DataFrame with ratings information
        output_dir: Directory to save files
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    movie_data.to_csv(output_path / 'movies.csv', index=False)
    ratings_data.to_csv(output_path / 'ratings.csv', index=False)
    
    print(f"Data saved to {output_dir}/movies.csv and {output_dir}/ratings.csv")


def get_popular_movies(ratings_data, movie_data, n=10):
    """
    Get most popular movies by average rating
    
    Args:
        ratings_data: DataFrame with ratings
        movie_data: DataFrame with movie information
        n: Number of popular movies to return
        
    Returns:
        DataFrame with popular movies
    """
    avg_ratings = ratings_data.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
    avg_ratings.columns = ['movieId', 'avg_rating', 'num_ratings']
    
    # Filter movies with at least 5 ratings
    avg_ratings = avg_ratings[avg_ratings['num_ratings'] >= 5]
    avg_ratings = avg_ratings.sort_values(['avg_rating', 'num_ratings'], ascending=False)
    
    popular = avg_ratings.merge(movie_data, on='movieId', how='left')
    return popular[['movieId', 'title', 'genres', 'avg_rating', 'num_ratings']].head(n)

