# Movie Recommendation System

A comprehensive movie recommendation system built with Python that implements multiple recommendation algorithms including content-based filtering, collaborative filtering, and hybrid approaches.

## Features

- **Content-Based Filtering**: Recommends movies similar to a given movie based on genre and description features
- **Collaborative Filtering**: Recommends movies based on user behavior and ratings patterns
- **Hybrid Approach**: Combines both content-based and collaborative filtering for better recommendations
- **User-Based Recommendations**: Get personalized recommendations for specific users
- **Popular Movies**: Discover most popular movies by average ratings

## Installation

1. Clone or download this repository

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Streamlit Web Interface (Recommended)

Launch the interactive web interface:

```bash
streamlit run streamlit_app.py
```

The app will open in your default web browser with a user-friendly interface where you can:
- Get hybrid recommendations (combines content-based + collaborative filtering)
- Get content-based recommendations
- Get collaborative filtering recommendations
- Get user-based personalized recommendations
- Browse popular movies

### Command Line Interface

#### 1. Get Hybrid Recommendations (Recommended)
```bash
python main.py --mode hybrid --movie "The Dark Knight" --n 10
```

#### 2. Content-Based Recommendations
```bash
python main.py --mode content --movie "Inception" --n 10
```

#### 3. Collaborative Filtering Recommendations
```bash
python main.py --mode collaborative --movie "Pulp Fiction" --n 10
```

#### 4. User-Based Recommendations
```bash
python main.py --mode user --user 1 --n 10
```

#### 5. List All Available Movies
```bash
python main.py --mode list
```

#### 6. Get Popular Movies
```bash
python main.py --mode popular --n 10
```

### Using Your Own Data

If you have your own movie and ratings data in CSV format:

```bash
python main.py --mode hybrid --movie "Your Movie Title" --movies-file data/movies.csv --ratings-file data/ratings.csv
```

**Required CSV Formats:**

1. **movies.csv** should contain:
   - `movieId`: Unique movie identifier
   - `title`: Movie title
   - `genres`: Pipe-separated genres (e.g., "Action|Adventure|Sci-Fi")
   - `description`: (Optional) Movie description

2. **ratings.csv** should contain:
   - `userId`: User identifier
   - `movieId`: Movie identifier (matches movies.csv)
   - `rating`: Rating value (typically 1-5)

### Python API

You can also use the recommendation engines directly in your Python code:

```python
from recommendation_engine import ContentBasedRecommender, CollaborativeFilteringRecommender, HybridRecommender
from data_utils import load_sample_data

# Load data
movie_data, ratings_data = load_sample_data()

# Content-Based Recommendations
content_recommender = ContentBasedRecommender()
content_recommender.fit(movie_data)
recommendations = content_recommender.recommend("The Dark Knight", n_recommendations=10)
print(recommendations)

# Collaborative Filtering
collab_recommender = CollaborativeFilteringRecommender()
collab_recommender.fit(ratings_data, movie_data)
recommendations = collab_recommender.recommend_similar_movies("Inception", n_recommendations=10)
print(recommendations)

# Hybrid Recommendations
hybrid_recommender = HybridRecommender()
hybrid_recommender.fit(movie_data, ratings_data)
recommendations = hybrid_recommender.recommend("The Matrix", n_recommendations=10)
print(recommendations)
```

## How It Works

### Content-Based Filtering
- Uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to convert movie genres/descriptions into numerical features
- Calculates cosine similarity between movies
- Recommends movies with similar feature vectors

### Collaborative Filtering
- Creates a user-item matrix from ratings data
- Uses K-Nearest Neighbors to find users with similar preferences
- Recommends movies liked by similar users

### Hybrid Approach
- Combines scores from both content-based and collaborative filtering
- Weighted combination provides more balanced recommendations
- Default weights: 50% content-based, 50% collaborative

## Sample Data

The system includes sample data with 50 movies and ratings from 100 users. You can use this for testing or replace it with your own dataset.

## Requirements

- Python 3.7+
- numpy
- pandas
- scikit-learn
- scipy
- streamlit (for web interface)

## Future Enhancements

- Deep learning-based recommendations
- Real-time recommendation updates
- More sophisticated feature engineering
- Integration with movie databases (TMDB, IMDb)
- User authentication and personal profiles
