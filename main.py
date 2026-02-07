"""
Main application for Movie Recommendation System
"""

import argparse
from recommendation_engine import (
    ContentBasedRecommender,
    CollaborativeFilteringRecommender,
    HybridRecommender
)
from data_utils import load_sample_data, load_data_from_files, get_popular_movies
import pandas as pd


def print_recommendations(recommendations, title="Recommendations"):
    """Pretty print recommendations"""
    print(f"\n{title}:")
    print("-" * 60)
    if not recommendations:
        print("No recommendations found.")
    else:
        for i, (movie, score) in enumerate(recommendations, 1):
            print(f"{i}. {movie} (score: {score:.3f})")
    print()


def content_based_recommendations(movie_data, movie_title, n=10):
    """Get content-based recommendations"""
    print(f"\n{'='*60}")
    print("Content-Based Recommendations")
    print(f"{'='*60}")
    
    recommender = ContentBasedRecommender()
    recommender.fit(movie_data)
    
    recommendations = recommender.recommend(movie_title, n)
    print_recommendations(recommendations, f"Movies similar to '{movie_title}'")
    
    return recommendations


def collaborative_recommendations(ratings_data, movie_data, movie_title, n=10):
    """Get collaborative filtering recommendations"""
    print(f"\n{'='*60}")
    print("Collaborative Filtering Recommendations")
    print(f"{'='*60}")
    
    recommender = CollaborativeFilteringRecommender()
    recommender.fit(ratings_data, movie_data)
    
    recommendations = recommender.recommend_similar_movies(movie_title, n)
    print_recommendations(recommendations, f"Movies recommended based on '{movie_title}'")
    
    return recommendations


def hybrid_recommendations(movie_data, ratings_data, movie_title, n=10):
    """Get hybrid recommendations"""
    print(f"\n{'='*60}")
    print("Hybrid Recommendations (Content-Based + Collaborative Filtering)")
    print(f"{'='*60}")
    
    recommender = HybridRecommender()
    recommender.fit(movie_data, ratings_data)
    
    recommendations = recommender.recommend(movie_title, n)
    print_recommendations(recommendations, f"Movies recommended based on '{movie_title}'")
    
    return recommendations


def user_recommendations(ratings_data, movie_data, user_id, n=10):
    """Get recommendations for a specific user"""
    print(f"\n{'='*60}")
    print("User-Based Recommendations")
    print(f"{'='*60}")
    
    recommender = CollaborativeFilteringRecommender()
    recommender.fit(ratings_data, movie_data)
    
    recommendations = recommender.recommend_for_user(user_id, n)
    print_recommendations(recommendations, f"Recommendations for User {user_id}")
    
    return recommendations


def list_movies(movie_data):
    """List all available movies"""
    print(f"\n{'='*60}")
    print("Available Movies")
    print(f"{'='*60}")
    for idx, row in movie_data.iterrows():
        print(f"{row['movieId']}. {row['title']} ({row['genres']})")
    print()


def main():
    parser = argparse.ArgumentParser(description='Movie Recommendation System')
    parser.add_argument(
        '--mode',
        choices=['content', 'collaborative', 'hybrid', 'user', 'list', 'popular'],
        default='hybrid',
        help='Recommendation mode'
    )
    parser.add_argument(
        '--movie',
        type=str,
        help='Movie title for recommendations'
    )
    parser.add_argument(
        '--user',
        type=int,
        help='User ID for user-based recommendations'
    )
    parser.add_argument(
        '--n',
        type=int,
        default=10,
        help='Number of recommendations (default: 10)'
    )
    parser.add_argument(
        '--movies-file',
        type=str,
        help='Path to movies CSV file'
    )
    parser.add_argument(
        '--ratings-file',
        type=str,
        help='Path to ratings CSV file'
    )
    
    args = parser.parse_args()
    
    # Load data
    if args.movies_file and args.ratings_file:
        try:
            movie_data, ratings_data = load_data_from_files(args.movies_file, args.ratings_file)
            print(f"Loaded data from files: {args.movies_file}, {args.ratings_file}")
        except Exception as e:
            print(f"Error loading files: {e}")
            print("Using sample data instead...")
            movie_data, ratings_data = load_sample_data()
    else:
        print("No data files specified. Using sample data...")
        movie_data, ratings_data = load_sample_data()
    
    print(f"\nLoaded {len(movie_data)} movies and {len(ratings_data)} ratings from {ratings_data['userId'].nunique()} users")
    
    # Execute based on mode
    if args.mode == 'list':
        list_movies(movie_data)
    
    elif args.mode == 'popular':
        popular = get_popular_movies(ratings_data, movie_data, args.n)
        print(f"\n{'='*60}")
        print("Most Popular Movies")
        print(f"{'='*60}")
        for idx, row in popular.iterrows():
            print(f"{row['title']} - Avg Rating: {row['avg_rating']:.2f} ({row['num_ratings']} ratings)")
        print()
    
    elif args.mode == 'user':
        if args.user is None:
            print("Error: --user option required for user-based recommendations")
            print("Try: python main.py --mode user --user 1")
            return
        
        user_recommendations(ratings_data, movie_data, args.user, args.n)
    
    else:
        if args.movie is None:
            print("Error: --movie option required for movie-based recommendations")
            print("Available movies:")
            print(movie_data['title'].head(10).tolist())
            print("\nTry: python main.py --mode hybrid --movie 'The Dark Knight'")
            return
        
        if args.mode == 'content':
            content_based_recommendations(movie_data, args.movie, args.n)
        elif args.mode == 'collaborative':
            collaborative_recommendations(ratings_data, movie_data, args.movie, args.n)
        elif args.mode == 'hybrid':
            hybrid_recommendations(movie_data, ratings_data, args.movie, args.n)


if __name__ == '__main__':
    main()

