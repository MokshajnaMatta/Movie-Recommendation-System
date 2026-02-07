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
