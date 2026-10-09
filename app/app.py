from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AI Course Recommender",
    page_icon="🎓",
    layout="wide"
)


# ==================================================
# DATA LOADING
# ==================================================

@st.cache_data
def load_data():

    course_genre_df = pd.read_csv(Path(__file__).parent / "course_genre.csv")

    course_processed_df = pd.read_csv(Path(__file__).parent / "course_processed.csv")

    ratings_df = pd.read_csv(Path(__file__).parent / "ratings.csv")

    return (
        course_genre_df,
        course_processed_df,
        ratings_df
    )


df1, df2, df3 = load_data()


# ==================================================
# DATA CLEANING
# ==================================================

df1 = df1.drop_duplicates(
    subset=["COURSE_ID"]
).copy()

df2 = df2.drop_duplicates(
    subset=["COURSE_ID"]
).copy()

df3 = df3.dropna(
    subset=["user", "item"]
).copy()


df2["TITLE"] = (
    df2["TITLE"]
    .fillna("")
    .astype(str)
)

df2["DESCRIPTION"] = (
    df2["DESCRIPTION"]
    .fillna("")
    .astype(str)
)


# ==================================================
# COURSE GENRE COLUMNS
# ==================================================

genre_columns = [
    "Database",
    "Python",
    "CloudComputing",
    "DataAnalysis",
    "Containers",
    "MachineLearning",
    "ComputerVision",
    "DataScience",
    "BigData",
    "Chatbot",
    "R",
    "BackendDev",
    "FrontendDev",
    "Blockchain"
]


# ==================================================
# COURSE TEXT
# ==================================================

df2["course_text"] = (
    df2["TITLE"] +
    " " +
    df2["DESCRIPTION"]
)


# ==================================================
# TF-IDF MODEL
# ==================================================

@st.cache_resource
def create_tfidf(text_data):

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=1000
    )

    matrix = vectorizer.fit_transform(
        text_data
    )

    return vectorizer, matrix


tfidf_vectorizer, tfidf_matrix = create_tfidf(
    df2["course_text"]
)


# ==================================================
# COURSE SIMILARITY
# ==================================================

def get_similar_courses(
    course_index,
    n=5
):

    scores = cosine_similarity(
        tfidf_matrix[course_index],
        tfidf_matrix
    ).flatten()

    similar_indices = scores.argsort()[::-1]

    similar_indices = similar_indices[
        similar_indices != course_index
    ][:n]

    return (
        similar_indices,
        scores[similar_indices]
    )


# ==================================================
# USER PROFILE CREATION
# ==================================================

user_genres = df3.merge(
    df1[
        ["COURSE_ID"] +
        genre_columns
    ],
    left_on="item",
    right_on="COURSE_ID",
    how="inner"
)


user_profiles = (
    user_genres
    .groupby("user")[genre_columns]
    .mean()
)


course_profiles = (
    df1
    .set_index("COURSE_ID")[genre_columns]
)


# ==================================================
# PERSONALIZED RECOMMENDATIONS
# ==================================================

def get_personalized_recommendations(
    user_id,
    n=5
):

    if user_id not in user_profiles.index:

        return pd.DataFrame()

    user_vector = (
        user_profiles
        .loc[user_id]
        .values
        .reshape(1, -1)
    )

    scores = cosine_similarity(
        user_vector,
        course_profiles.values
    ).flatten()

    top_indices = (
        scores
        .argsort()[::-1][:n]
    )

    recommended_ids = (
        course_profiles
        .index[
            top_indices
        ]
    )

    recommendations = df2[
        df2["COURSE_ID"].isin(
            recommended_ids
        )
    ].copy()

    score_map = dict(
        zip(
            recommended_ids,
            scores[top_indices]
        )
    )

    recommendations[
        "Similarity Score"
    ] = (
        recommendations[
            "COURSE_ID"
        ].map(score_map)
    )

    recommendations = (
        recommendations
        .sort_values(
            "Similarity Score",
            ascending=False
        )
    )

    return recommendations


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("🎓 AI Course Recommender")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Course Similarity",
        "Personalized Recommendations",
        "Course Information",
        "About"
    ]
)


# ==================================================
# HOME PAGE
# ==================================================

if page == "Home":

    st.title(
        "🎓 AI Course Recommender System"
    )

    st.markdown(
        """
        ### Personalized Learning Through Machine Learning

        This application recommends online courses using
        **course content, learner interests, and historical
        course interactions**.

        The system combines content-based recommendation
        techniques with learner-profile analysis to help
        users discover relevant courses more efficiently.
        """
    )

    st.markdown("---")

    st.subheader(
        "📊 Dataset Overview"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Courses",
            f"{df2['COURSE_ID'].nunique():,}"
        )

    with col2:

        st.metric(
            "Learners",
            f"{df3['user'].nunique():,}"
        )

    with col3:

        st.metric(
            "Interactions",
            f"{len(df3):,}"
        )

    with col4:

        st.metric(
            "Course Genres",
            len(genre_columns)
        )

    st.markdown("---")

    st.subheader(
        "🤖 Recommendation Approaches"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "### 🔎 Content-Based Recommendation"
        )

        st.write(
            """
            Course titles and descriptions are converted
            into **TF-IDF feature vectors**.

            **Cosine similarity** is then used to identify
            courses with similar content.
            """
        )

        st.info(
            "TF-IDF + Cosine Similarity"
        )

    with col2:

        st.markdown(
            "### 👤 Personalized Recommendation"
        )

        st.write(
            """
            Learner interactions are combined with course
            genre information to create a **learner profile**.

            The learner profile is compared with course
            genre vectors to identify relevant courses.
            """
        )

        st.info(
            "User Profiles + Course Genre Similarity"
        )

    st.markdown("---")

    st.subheader(
        "⚙️ Recommendation Workflow"
    )

    workflow = [
        "1️⃣ Load Course and Learner Data",
        "2️⃣ Clean and Preprocess Data",
        "3️⃣ Extract Course Features",
        "4️⃣ Build Learner Profiles",
        "5️⃣ Calculate Similarity",
        "6️⃣ Generate Recommendations"
    ]

    for step in workflow:

        st.write(step)

    st.markdown("---")

    st.subheader(
        "🎯 Project Goal"
    )

    st.write(
        """
        The goal of this project is to demonstrate how
        machine learning can be applied to online learning
        platforms to improve course discovery and provide
        more personalized learning recommendations.
        """
    )


# ==================================================
# COURSE SIMILARITY PAGE
# ==================================================

elif page == "Course Similarity":

    st.header(
        "🔎 Find Similar Courses"
    )

    st.write(
        "Select a course to find other courses with similar content."
    )

    course_options = (
        df2["TITLE"]
        .drop_duplicates()
        .tolist()
    )

    selected_title = st.selectbox(
        "Select a course",
        course_options
    )

    selected_rows = df2[
        df2["TITLE"] == selected_title
    ]

    if not selected_rows.empty:

        selected_row = (
            selected_rows.iloc[0]
        )

        selected_index = (
            df2.index[
                df2["TITLE"] ==
                selected_title
            ][0]
        )

        st.markdown("---")

        st.subheader(
            "📚 Selected Course"
        )

        st.markdown(
            f"### {selected_title}"
        )

        if selected_row["DESCRIPTION"]:

            st.write(
                selected_row["DESCRIPTION"]
            )

        st.markdown("---")

        if st.button(
            "🔎 Recommend Similar Courses"
        ):

            indices, scores = (
                get_similar_courses(
                    selected_index,
                    n=5
                )
            )

            st.subheader(
                "🎓 Recommended Similar Courses"
            )

            for rank, (
                index,
                score
            ) in enumerate(
                zip(indices, scores),
                start=1
            ):

                title = (
                    df2
                    .iloc[index]["TITLE"]
                )

                description = (
                    df2
                    .iloc[index]["DESCRIPTION"]
                )

                st.markdown(
                    f"### {rank}. {title}"
                )

                st.progress(
                    float(score)
                )

                st.caption(
                    f"Similarity Score: {score:.3f}"
                )

                if description:

                    st.write(
                        description
                    )

                st.markdown("---")


# ==================================================
# PERSONALIZED RECOMMENDATIONS PAGE
# ==================================================

elif page == "Personalized Recommendations":

    st.header(
        "👤 Personalized Course Recommendations"
    )

    st.write(
        """
        Select a learner to generate course recommendations
        based on their historical course interests.
        """
    )

    user_options = list(
        user_profiles.index
    )

    selected_user = st.selectbox(
        "Select a learner",
        user_options
    )

    # ----------------------------------------------
    # LEARNER PROFILE
    # ----------------------------------------------

    st.markdown("---")

    st.subheader(
        "📊 Learner Interest Profile"
    )

    user_profile = (
        user_profiles
        .loc[selected_user]
    )

    profile_df = (
        user_profile
        .sort_values(
            ascending=False
        )
        .reset_index()
    )

    profile_df.columns = [
        "Course Genre",
        "Preference Score"
    ]

    profile_df = profile_df.head(5)

    st.dataframe(
        profile_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # ----------------------------------------------
    # RECOMMENDATIONS
    # ----------------------------------------------

    if st.button(
        "🎯 Generate Personalized Recommendations"
    ):

        recommendations = (
            get_personalized_recommendations(
                selected_user,
                n=5
            )
        )

        if recommendations.empty:

            st.warning(
                "No recommendations available for this learner."
            )

        else:

            st.subheader(
                "🎓 Recommended Courses"
            )

            for rank, (
                _,
                row
            ) in enumerate(
                recommendations.iterrows(),
                start=1
            ):

                st.markdown(
                    f"### {rank}. {row['TITLE']}"
                )

                st.caption(
                    "Profile Similarity Score: "
                    f"{row['Similarity Score']:.3f}"
                )

                if row["DESCRIPTION"]:

                    st.write(
                        row["DESCRIPTION"]
                    )

                st.markdown("---")


# ==================================================
# COURSE INFORMATION PAGE
# ==================================================

elif page == "Course Information":

    st.header(
        "📚 Course Information"
    )

    selected_course = st.selectbox(
        "Select a course",
        df2["TITLE"].tolist()
    )

    selected_rows = df2[
        df2["TITLE"] ==
        selected_course
    ]

    if not selected_rows.empty:

        course = (
            selected_rows.iloc[0]
        )

        st.subheader(
            course["TITLE"]
        )

        st.write(
            course["DESCRIPTION"]
        )

        course_id = (
            course["COURSE_ID"]
        )

        genre_row = df1[
            df1["COURSE_ID"] ==
            course_id
        ]

        if not genre_row.empty:

            available_genres = []

            for genre in genre_columns:

                if genre in genre_row.columns:

                    value = (
                        genre_row
                        .iloc[0][genre]
                    )

                    if value == 1:

                        available_genres.append(
                            genre
                        )

            if available_genres:

                st.subheader(
                    "🏷️ Course Genres"
                )

                st.write(
                    ", ".join(
                        available_genres
                    )
                )


# ==================================================
# ABOUT PAGE
# ==================================================

elif page == "About":

    st.header(
        "ℹ️ About the Project"
    )

    st.markdown(
        """
        ## AI Course Recommender System

        This project explores machine learning techniques
        for personalized online course recommendation.

        ### Techniques Used

        - TF-IDF
        - Cosine Similarity
        - Course Genre Features
        - User Profile Modeling
        - Content-Based Recommendation
        - Collaborative Filtering

        ### Project Purpose

        The system is designed to help learners discover
        relevant online courses by analyzing course content
        and learner preferences.

        ### Technology

        - Python
        - Pandas
        - NumPy
        - Scikit-learn
        - Streamlit

        The project was developed as part of the
        **IBM Machine Learning Professional Certificate**.
        """
    )