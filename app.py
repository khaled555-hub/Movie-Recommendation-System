# =====================================
# Hybrid Movie Recommendation System
# Streamlit Application
# =====================================

# =====================================
# IMPORT LIBRARIES
# =====================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from surprise import Dataset
from surprise import Reader
from surprise import SVD

# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="Hybrid Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# =====================================
# TITLE
# =====================================

st.title("🎬 Hybrid Movie Recommendation System")

st.write(
    """
    This system combines:
    - Content-Based Filtering
    - Collaborative Filtering (Surprise SVD)
    - Hybrid Recommendation Engine
    """
)

# =====================================
# LOAD DATA
# =====================================

@st.cache_data
def load_data():

    movies = pd.read_csv(
        "ml-latest-small/movies.csv"
    )

    ratings = pd.read_csv(
        "ml-latest-small/ratings.csv"
    )

    tags = pd.read_csv(
        "ml-latest-small/tags.csv"
    )

    return movies, ratings, tags

movies, ratings, tags = load_data()

# =====================================
# DATA CLEANING
# =====================================

movies = movies.drop_duplicates()
ratings = ratings.drop_duplicates()

movies['genres'] = movies[
    'genres'
].str.replace('|', ' ')

movies['title'] = movies[
    'title'
].str.strip()

movies = movies[
    movies['genres']
    != '(no genres listed)'
]

# =====================================
# MERGE TAGS
# =====================================

tags_grouped = tags.groupby(
    'movieId'
)['tag'].apply(
    lambda x: ' '.join(
        x.astype(str)
    )
)

movies = movies.merge(
    tags_grouped,
    on='movieId',
    how='left'
)

movies['tag'] = movies[
    'tag'
].fillna('')

# =====================================
# CREATE METADATA
# =====================================

movies['metadata'] = (
    movies['genres']
    + ' '
    + movies['tag']
)

# =====================================
# TF-IDF VECTORIZATION
# =====================================

@st.cache_resource
def create_tfidf(metadata):

    tfidf = TfidfVectorizer(
        stop_words='english'
    )

    tfidf_matrix = tfidf.fit_transform(
        metadata
    )

    return tfidf, tfidf_matrix

tfidf, tfidf_matrix = create_tfidf(
    movies['metadata']
)

# =====================================
# MOVIE INDICES
# =====================================

indices = pd.Series(
    movies.index,
    index=movies['title']
)

indices = indices[
    ~indices.index.duplicated(
        keep='first'
    )
]

# =====================================
# TRAIN SURPRISE SVD MODEL
# =====================================

@st.cache_resource
def train_svd_model(ratings):

    reader = Reader(
        rating_scale=(0.5, 5)
    )

    data = Dataset.load_from_df(
        ratings[
            ['userId', 'movieId', 'rating']
        ],
        reader
    )

    trainset = data.build_full_trainset()

    svd_model = SVD(
        n_factors=100,
        random_state=42
    )

    svd_model.fit(trainset)

    return svd_model

svd_model = train_svd_model(ratings)

# =====================================
# HYBRID RECOMMENDATION FUNCTION
# =====================================

def hybrid_recommendations(
    user_id,
    title,
    top_n=10,
    alpha=0.5
):

    # Get selected movie index
    idx = indices[title]

    # =================================
    # Dynamic Cosine Similarity
    # =================================

    sim_scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    # Enumerate similarity scores
    sim_scores = list(
        enumerate(sim_scores)
    )

    # Sort by similarity
    sim_scores = sorted(
        sim_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Remove selected movie itself
    sim_scores = sim_scores[1:50]

    # Candidate movie indices
    movie_indices = [
        i[0] for i in sim_scores
    ]

    # Candidate movies
    candidate_movies = movies.iloc[
        movie_indices
    ][[
        'movieId',
        'title',
        'genres'
    ]]

    # Store hybrid scores
    hybrid_scores = []

    # =================================
    # Calculate Hybrid Scores
    # =================================

    for movie_index, row in zip(
        movie_indices,
        candidate_movies.iterrows()
    ):

        row = row[1]

        movie_id = row['movieId']

        # -----------------------------
        # Content-Based Score
        # -----------------------------

        content_score = cosine_similarity(
            tfidf_matrix[idx],
            tfidf_matrix[movie_index]
        )[0][0]

        # -----------------------------
        # Collaborative Score
        # -----------------------------

        try:

            collaborative_score = svd_model.predict(
                user_id,
                movie_id
            ).est

            collaborative_score = (
                collaborative_score / 5
            )

        except:

            collaborative_score = 0

        # -----------------------------
        # Hybrid Score
        # -----------------------------

        hybrid_score = (
            alpha * content_score
            +
            (1 - alpha)
            * collaborative_score
        )

        hybrid_scores.append(
            hybrid_score
        )

    # Add scores
    candidate_movies[
        'hybrid_score'
    ] = hybrid_scores

    # Sort recommendations
    recommendations = candidate_movies.sort_values(
        by='hybrid_score',
        ascending=False
    )

    return recommendations.head(top_n)

# =====================================
# SIDEBAR
# =====================================

st.sidebar.header("Recommendation Settings")

user_id = st.sidebar.number_input(
    "Enter User ID",
    min_value=1,
    max_value=610,
    value=1
)

movie_title = st.sidebar.selectbox(
    "Choose Your Favorite Movie",
    sorted(
        movies['title'].unique()
    )
)

number_of_recommendations = st.sidebar.slider(
    "Number of Recommendations",
    min_value=5,
    max_value=20,
    value=10
)

alpha = st.sidebar.slider(
    "Hybrid Weight",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.1
)

# =====================================
# RECOMMENDATION BUTTON
# =====================================

if st.button("Recommend Movies"):

    recommendations = hybrid_recommendations(
        user_id=user_id,
        title=movie_title,
        top_n=number_of_recommendations,
        alpha=alpha
    )

    st.subheader(
        "🎥 Recommended Movies"
    )

    st.dataframe(
        recommendations[
            [
                'title',
                'genres',
                'hybrid_score'
            ]
        ],
        use_container_width=True
    )

# =====================================
# FOOTER
# =====================================

st.markdown("---")

st.write(
    "Developed using Streamlit, TF-IDF, Cosine Similarity, and Surprise SVD"
)

# =====================================
# RUN COMMAND
# =====================================

# streamlit run app.py