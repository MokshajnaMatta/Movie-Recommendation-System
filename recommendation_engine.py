"""
Movie Recommendation Engine
Supports both content-based and collaborative filtering approaches
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import pandas as pd


class ContentBasedRecommender:
    """Content-based recommendation using movie features"""
    
    def __init__(self):
        self.movie_data = None
        self.vectorizer = None
        self.feature_matrix = None
        
    def fit(self, movie_data):
        """
        Train the content-based recommender
        
        Args:
            movie_data: DataFrame with columns: movieId, title, genres, description (optional)
        """
        self.movie_data = movie_data.copy()
        
        # Combine genres and description for feature extraction
        if 'description' in movie_data.columns:
            self.movie_data['combined_features'] = (
                movie_data['genres'].fillna('') + ' ' + 
                movie_data['description'].fillna('')
            )
        else:
            self.movie_data['combined_features'] = movie_data['genres'].fillna('')
        
        # Create TF-IDF matrix
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.feature_matrix = self.vectorizer.fit_transform(self.movie_data['combined_features'])
        
    def recommend(self, movie_title, n_recommendations=10):
        """
        Get recommendations for a movie
        
        Args:
            movie_title: Title of the movie
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of recommended movie titles
        """
        if self.feature_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Find the movie index
        movie_idx = self.movie_data[self.movie_data['title'].str.lower() == movie_title.lower()].index
        
        if len(movie_idx) == 0:
            return []
        
        movie_idx = movie_idx[0]
        
        # Calculate cosine similarity
        similarity_scores = cosine_similarity(
            self.feature_matrix[movie_idx],
            self.feature_matrix
        ).flatten()
        
        # Get top similar movies (excluding the movie itself)
        similar_indices = similarity_scores.argsort()[::-1][1:n_recommendations+1]
        recommendations = self.movie_data.iloc[similar_indices]['title'].tolist()
        scores = similarity_scores[similar_indices]
        
        return list(zip(recommendations, scores))


class CollaborativeFilteringRecommender:
    """Collaborative filtering using user-item matrix"""
    
    def __init__(self, n_neighbors=10):
        self.ratings_data = None
        self.movie_data = None
        self.user_movie_matrix = None
        self.model_knn = None
        self.n_neighbors = n_neighbors
        
    def fit(self, ratings_data, movie_data):
        """
        Train the collaborative filtering recommender
        
        Args:
            ratings_data: DataFrame with columns: userId, movieId, rating
            movie_data: DataFrame with columns: movieId, title
        """
        self.ratings_data = ratings_data.copy()
        self.movie_data = movie_data.copy()
        
        # Create user-item matrix
        user_movie_pivot = ratings_data.pivot(
            index='userId',
            columns='movieId',
            values='rating'
        ).fillna(0)
        
        # Convert to sparse matrix for efficiency
        self.user_movie_matrix = csr_matrix(user_movie_pivot.values)
        
        # Train KNN model
        self.model_knn = NearestNeighbors(
            metric='cosine',
            algorithm='brute',
            n_neighbors=self.n_neighbors + 1,
            n_jobs=-1
        )
        self.model_knn.fit(self.user_movie_matrix)
        
        # Store user and movie mappings
        self.user_ids = user_movie_pivot.index
        self.movie_ids = user_movie_pivot.columns
        
    def recommend_for_user(self, user_id, n_recommendations=10):
        """
        Get recommendations for a user
        
        Args:
            user_id: ID of the user
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of recommended movie titles
        """
        if self.model_knn is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if user_id not in self.user_ids.values:
            return []
        
        user_idx = np.where(self.user_ids == user_id)[0][0]
        
        # Find similar users
        distances, indices = self.model_knn.kneighbors(
            self.user_movie_matrix[user_idx],
            n_neighbors=self.n_neighbors + 1
        )
        
        # Get movies rated by similar users
        similar_users_ratings = self.user_movie_matrix[indices[0][1:]].toarray()
        
        # Calculate weighted average ratings
        user_ratings = self.user_movie_matrix[user_idx].toarray().flatten()
        weighted_ratings = np.zeros(len(self.movie_ids))
        
        for i, similar_user_ratings in enumerate(similar_users_ratings):
            weight = 1 - distances[0][i+1]  # Closer users have higher weight
            weighted_ratings += weight * similar_user_ratings
        
        # Get top recommendations (excluding already rated movies)
        recommendations_idx = np.argsort(weighted_ratings)[::-1]
        recommendations = []
        
        for idx in recommendations_idx:
            if user_ratings[idx] == 0 and len(recommendations) < n_recommendations:
                movie_id = self.movie_ids[idx]
                movie_title = self.movie_data[self.movie_data['movieId'] == movie_id]['title'].values
                if len(movie_title) > 0:
                    recommendations.append((movie_title[0], weighted_ratings[idx]))
        
        return recommendations
    
    def recommend_similar_movies(self, movie_title, n_recommendations=10):
        """
        Get recommendations based on movies similar users liked
        
        Args:
            movie_title: Title of the movie
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of recommended movie titles
        """
        if self.model_knn is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Find movie ID
        movie_match = self.movie_data[self.movie_data['title'].str.lower() == movie_title.lower()]
        if len(movie_match) == 0:
            return []
        
        movie_id = movie_match.iloc[0]['movieId']
        if movie_id not in self.movie_ids.values:
            return []
        
        movie_idx = np.where(self.movie_ids == movie_id)[0][0]
        
        # Find users who rated this movie highly
        movie_ratings = self.user_movie_matrix[:, movie_idx].toarray().flatten()
        high_rated_users = np.where(movie_ratings >= 4.0)[0]
        
        if len(high_rated_users) == 0:
            return []
        
        # Aggregate ratings from users who liked this movie
        aggregated_ratings = np.zeros(len(self.movie_ids))
        
        for user_idx in high_rated_users:
            user_ratings = self.user_movie_matrix[user_idx].toarray().flatten()
            aggregated_ratings += user_ratings
        
        # Get top recommendations (excluding the input movie)
        recommendations_idx = np.argsort(aggregated_ratings)[::-1]
        recommendations = []
        
        for idx in recommendations_idx:
            if idx != movie_idx and len(recommendations) < n_recommendations:
                rec_movie_id = self.movie_ids[idx]
                movie_title = self.movie_data[self.movie_data['movieId'] == rec_movie_id]['title'].values
                if len(movie_title) > 0 and aggregated_ratings[idx] > 0:
                    recommendations.append((movie_title[0], aggregated_ratings[idx]))
        
        return recommendations


class HybridRecommender:
    """Hybrid recommendation combining content-based and collaborative filtering"""
    
    def __init__(self, content_weight=0.5, collaborative_weight=0.5):
        self.content_recommender = ContentBasedRecommender()
        self.collaborative_recommender = CollaborativeFilteringRecommender()
        self.content_weight = content_weight
        self.collaborative_weight = collaborative_weight
        self.movie_data = None
        
    def fit(self, movie_data, ratings_data):
        """
        Train both recommenders
        
        Args:
            movie_data: DataFrame with movie information
            ratings_data: DataFrame with ratings information
        """
        self.movie_data = movie_data.copy()
        self.content_recommender.fit(movie_data)
        self.collaborative_recommender.fit(ratings_data, movie_data)
        
    def recommend(self, movie_title, n_recommendations=10):
        """
        Get hybrid recommendations
        
        Args:
            movie_title: Title of the movie
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of recommended movie titles
        """
        # Get recommendations from both approaches
        content_recs = dict(self.content_recommender.recommend(movie_title, n_recommendations * 2))
        collaborative_recs = dict(self.collaborative_recommender.recommend_similar_movies(
            movie_title, n_recommendations * 2
        ))
        
        # Normalize scores
        if content_recs:
            max_content_score = max(content_recs.values())
            content_recs = {k: v / max_content_score for k, v in content_recs.items()}
        
        if collaborative_recs:
            max_collab_score = max(collaborative_recs.values())
            collaborative_recs = {k: v / max_collab_score for k, v in collaborative_recs.items()}
        
        # Combine scores
        combined_scores = {}
        all_movies = set(content_recs.keys()) | set(collaborative_recs.keys())
        
        for movie in all_movies:
            content_score = content_recs.get(movie, 0) * self.content_weight
            collab_score = collaborative_recs.get(movie, 0) * self.collaborative_weight
            combined_scores[movie] = content_score + collab_score
        
        # Sort and return top recommendations
        sorted_recommendations = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n_recommendations]
        
        return sorted_recommendations

