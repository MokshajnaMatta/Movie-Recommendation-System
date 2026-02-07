"""
Example usage of the Movie Recommendation System
"""

from recommendation_engine import (
    ContentBasedRecommender,
    CollaborativeFilteringRecommender,
    HybridRecommender
)
from data_utils import load_sample_data, get_popular_movies


def main():
    print("=" * 70)
    print("Movie Recommendation System - Example Usage")
    print("=" * 70)
    
    # Load sample data
    print("\n1. Loading sample data...")
    movie_data, ratings_data = load_sample_data()
    print(f"   [OK] Loaded {len(movie_data)} movies")
    print(f"   [OK] Loaded {len(ratings_data)} ratings from {ratings_data['userId'].nunique()} users")
    
    # Show popular movies
    print("\n2. Most Popular Movies:")
    print("-" * 70)
    popular = get_popular_movies(ratings_data, movie_data, n=5)
    for idx, row in popular.iterrows():
        print(f"   • {row['title']:40} Rating: {row['avg_rating']:.2f} ({row['num_ratings']} votes)")
    
    # Content-Based Recommendations
    print("\n3. Content-Based Recommendations for 'The Dark Knight':")
    print("-" * 70)
    content_recommender = ContentBasedRecommender()
    content_recommender.fit(movie_data)
    content_recs = content_recommender.recommend("The Dark Knight", n_recommendations=5)
    for i, (movie, score) in enumerate(content_recs, 1):
        print(f"   {i}. {movie:50} Similarity: {score:.3f}")
    
    # Collaborative Filtering Recommendations
    print("\n4. Collaborative Filtering Recommendations for 'Inception':")
    print("-" * 70)
    collab_recommender = CollaborativeFilteringRecommender()
    collab_recommender.fit(ratings_data, movie_data)
    collab_recs = collab_recommender.recommend_similar_movies("Inception", n_recommendations=5)
    for i, (movie, score) in enumerate(collab_recs, 1):
        print(f"   {i}. {movie:50} Score: {score:.2f}")
    
    # User-Based Recommendations
    print("\n5. User-Based Recommendations for User ID 1:")
    print("-" * 70)
    user_recs = collab_recommender.recommend_for_user(1, n_recommendations=5)
    for i, (movie, score) in enumerate(user_recs, 1):
        print(f"   {i}. {movie:50} Score: {score:.2f}")
    
    # Hybrid Recommendations
    print("\n6. Hybrid Recommendations for 'The Matrix':")
    print("-" * 70)
    hybrid_recommender = HybridRecommender()
    hybrid_recommender.fit(movie_data, ratings_data)
    hybrid_recs = hybrid_recommender.recommend("The Matrix", n_recommendations=5)
    for i, (movie, score) in enumerate(hybrid_recs, 1):
        print(f"   {i}. {movie:50} Combined Score: {score:.3f}")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)
    print("\nTo use the CLI, try:")
    print("  python main.py --mode hybrid --movie 'The Dark Knight'")
    print("  python main.py --mode user --user 1")


if __name__ == '__main__':
    main()

