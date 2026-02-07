"""
Streamlit Frontend for Movie Recommendation System
"""

import streamlit as st
import pandas as pd
from recommendation_engine import (
    ContentBasedRecommender,
    CollaborativeFilteringRecommender,
    HybridRecommender
)
from data_utils import load_sample_data, load_data_from_files, get_popular_movies


# Page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    .recommendation-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load data with caching"""
    try:
        movie_data, ratings_data = load_sample_data()
        return movie_data, ratings_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None


def main():
    # Header
    st.markdown('<p class="main-header">🎬 Movie Recommendation System</p>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading movie data..."):
        movie_data, ratings_data = load_data()
    
    if movie_data is None or ratings_data is None:
        st.error("Failed to load data. Please check your data files.")
        return
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Mode selection
    mode = st.sidebar.radio(
        "Recommendation Mode",
        ["Hybrid (Recommended)", "Content-Based", "Collaborative Filtering", "User-Based", "Popular Movies"]
    )
    
    # Initialize session state for recommenders
    if 'content_recommender' not in st.session_state:
        with st.spinner("Training content-based recommender..."):
            st.session_state.content_recommender = ContentBasedRecommender()
            st.session_state.content_recommender.fit(movie_data)
    
    if 'collab_recommender' not in st.session_state:
        with st.spinner("Training collaborative filtering recommender..."):
            st.session_state.collab_recommender = CollaborativeFilteringRecommender()
            st.session_state.collab_recommender.fit(ratings_data, movie_data)
    
    if 'hybrid_recommender' not in st.session_state:
        with st.spinner("Training hybrid recommender..."):
            st.session_state.hybrid_recommender = HybridRecommender()
            st.session_state.hybrid_recommender.fit(movie_data, ratings_data)
    
    # Main content area
    if mode == "Hybrid (Recommended)":
        show_hybrid_recommendations(movie_data, ratings_data)
    elif mode == "Content-Based":
        show_content_based_recommendations(movie_data)
    elif mode == "Collaborative Filtering":
        show_collaborative_recommendations(movie_data, ratings_data)
    elif mode == "User-Based":
        show_user_recommendations(movie_data, ratings_data)
    elif mode == "Popular Movies":
        show_popular_movies(movie_data, ratings_data)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Statistics")
    st.sidebar.metric("Total Movies", len(movie_data))
    st.sidebar.metric("Total Ratings", len(ratings_data))
    st.sidebar.metric("Total Users", ratings_data['userId'].nunique())


def show_hybrid_recommendations(movie_data, ratings_data):
    """Display hybrid recommendations"""
    st.header("🔀 Hybrid Recommendations")
    st.markdown("Get recommendations using both content-based and collaborative filtering approaches.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        movie_title = st.selectbox(
            "Select a movie",
            options=sorted(movie_data['title'].tolist()),
            key="hybrid_movie"
        )
    
    with col2:
        n_recommendations = st.slider("Number of recommendations", 5, 20, 10, key="hybrid_n")
    
    if st.button("Get Recommendations", key="hybrid_btn", type="primary"):
        with st.spinner("Finding recommendations..."):
            recommendations = st.session_state.hybrid_recommender.recommend(
                movie_title, n_recommendations
            )
        
        if recommendations:
            st.success(f"Found {len(recommendations)} recommendations based on '{movie_title}'")
            display_recommendations(recommendations, show_scores=True)
        else:
            st.warning("No recommendations found. Try a different movie.")


def show_content_based_recommendations(movie_data):
    """Display content-based recommendations"""
    st.header("📝 Content-Based Recommendations")
    st.markdown("Find movies similar to your selection based on genres and descriptions.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        movie_title = st.selectbox(
            "Select a movie",
            options=sorted(movie_data['title'].tolist()),
            key="content_movie"
        )
    
    with col2:
        n_recommendations = st.slider("Number of recommendations", 5, 20, 10, key="content_n")
    
    if st.button("Get Recommendations", key="content_btn", type="primary"):
        with st.spinner("Finding similar movies..."):
            recommendations = st.session_state.content_recommender.recommend(
                movie_title, n_recommendations
            )
        
        if recommendations:
            st.success(f"Found {len(recommendations)} similar movies to '{movie_title}'")
            display_recommendations(recommendations, show_scores=True)
        else:
            st.warning("No recommendations found. Try a different movie.")


def show_collaborative_recommendations(movie_data, ratings_data):
    """Display collaborative filtering recommendations"""
    st.header("👥 Collaborative Filtering Recommendations")
    st.markdown("Get recommendations based on what similar users liked.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        movie_title = st.selectbox(
            "Select a movie",
            options=sorted(movie_data['title'].tolist()),
            key="collab_movie"
        )
    
    with col2:
        n_recommendations = st.slider("Number of recommendations", 5, 20, 10, key="collab_n")
    
    if st.button("Get Recommendations", key="collab_btn", type="primary"):
        with st.spinner("Finding recommendations based on user preferences..."):
            recommendations = st.session_state.collab_recommender.recommend_similar_movies(
                movie_title, n_recommendations
            )
        
        if recommendations:
            st.success(f"Found {len(recommendations)} recommendations based on '{movie_title}'")
            display_recommendations(recommendations, show_scores=True)
        else:
            st.warning("No recommendations found. Try a different movie.")


def show_user_recommendations(movie_data, ratings_data):
    """Display user-based recommendations"""
    st.header("👤 User-Based Recommendations")
    st.markdown("Get personalized recommendations for a specific user.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_id = st.selectbox(
            "Select a user ID",
            options=sorted(ratings_data['userId'].unique()),
            key="user_id"
        )
    
    with col2:
        n_recommendations = st.slider("Number of recommendations", 5, 20, 10, key="user_n")
    
    # Show user's rated movies
    user_ratings = ratings_data[ratings_data['userId'] == user_id].merge(
        movie_data, on='movieId', how='left'
    )
    
    if len(user_ratings) > 0:
        with st.expander(f"📋 View User {user_id}'s Rated Movies"):
            display_user_ratings(user_ratings)
    
    if st.button("Get Recommendations", key="user_btn", type="primary"):
        with st.spinner(f"Finding recommendations for User {user_id}..."):
            recommendations = st.session_state.collab_recommender.recommend_for_user(
                user_id, n_recommendations
            )
        
        if recommendations:
            st.success(f"Found {len(recommendations)} recommendations for User {user_id}")
            display_recommendations(recommendations, show_scores=True)
        else:
            st.warning("No recommendations found for this user.")


def show_popular_movies(movie_data, ratings_data):
    """Display popular movies"""
    st.header("⭐ Popular Movies")
    st.markdown("Discover the most popular movies based on average ratings.")
    
    n_movies = st.slider("Number of movies to show", 5, 50, 10, key="popular_n")
    
    if st.button("Show Popular Movies", key="popular_btn", type="primary"):
        with st.spinner("Calculating popular movies..."):
            popular = get_popular_movies(ratings_data, movie_data, n_movies)
        
        if len(popular) > 0:
            st.success(f"Showing top {len(popular)} popular movies")
            display_popular_movies(popular)
        else:
            st.warning("No popular movies found.")


def display_recommendations(recommendations, show_scores=True):
    """Display recommendations in a nice format"""
    if not recommendations:
        return
    
    for i, (movie, score) in enumerate(recommendations, 1):
        with st.container():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{i}. {movie}**")
            with col2:
                if show_scores:
                    st.metric("Score", f"{score:.3f}")
            st.divider()


def display_user_ratings(user_ratings):
    """Display user's movie ratings"""
    df_display = user_ratings[['title', 'rating', 'genres']].sort_values('rating', ascending=False)
    df_display.columns = ['Movie Title', 'Rating', 'Genres']
    st.dataframe(df_display, use_container_width=True, hide_index=True)


def display_popular_movies(popular_df):
    """Display popular movies in a nice format"""
    st.dataframe(
        popular_df[['title', 'avg_rating', 'num_ratings', 'genres']],
        column_config={
            "title": "Movie Title",
            "avg_rating": st.column_config.NumberColumn(
                "Average Rating",
                format="%.2f ⭐",
                help="Average rating from all users"
            ),
            "num_ratings": st.column_config.NumberColumn(
                "Number of Ratings",
                format="%d votes",
                help="Total number of ratings"
            ),
            "genres": "Genres"
        },
        use_container_width=True,
        hide_index=True
    )


if __name__ == "__main__":
    main()

