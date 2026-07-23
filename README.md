# 🎬 Movie Recommendation System

A machine learning model that recommends movies similar to a given title, built using **collaborative filtering** with **k-Nearest Neighbors (kNN)** on the MovieLens dataset.

## Overview

This project builds a movie recommendation engine based on **item-based collaborative filtering**. It uses historical user ratings to identify movies with similar rating patterns, then recommends the most similar titles to a movie the user already likes — the same core idea behind "Users who liked this also liked..." systems.

## How It Works

1. **Data loading** — Movie metadata (`movies.csv`) and user ratings (`ratings.csv`) from the [MovieLens](https://grouplens.org/datasets/movielens/) dataset are loaded with `pandas`.
2. **Sparse matrix construction** — User and movie IDs are mapped to continuous indices, and a sparse **movie × user** ratings matrix is built with `scipy.sparse.csr_matrix` for memory-efficient storage of the mostly-empty ratings grid.
3. **Similarity search** — A `NearestNeighbors` model (scikit-learn) is fit on the sparse matrix using **cosine similarity** to find movies with the closest rating patterns to a target movie.
4. **Recommendation** — Given a movie title (partial matches supported), the model looks up its nearest neighbors and returns the top-N most similar movies.

## Tech Stack

- **Python**
- **pandas** & **NumPy** — data loading and manipulation
- **SciPy** (`csr_matrix`) — sparse matrix representation
- **scikit-learn** (`NearestNeighbors`) — cosine-similarity-based recommendation
- **Matplotlib** & **Seaborn** — exploratory data visualization
- **Jupyter Notebook** (Google Colab)

## Project Structure

```
Movie-Recommendation-System/
├── karthik_movie_recommendation.ipynb   # Main notebook: data prep, model, recommendations
├── movies.csv                           # Movie metadata (movieId, title, genres)
└── G.karthik reddy Major project internship.pdf  # Project report
```

> **Note:** `ratings.csv` (MovieLens user ratings) is required to run the notebook but is not included in this repo due to size — download it from [MovieLens](https://grouplens.org/datasets/movielens/) and place it alongside `movies.csv`.

## Getting Started

### Prerequisites

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn
```

### Running the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/karthik-0211/Movie-Recommendation-System.git
   cd Movie-Recommendation-System
   ```
2. Download the [MovieLens](https://grouplens.org/datasets/movielens/) `ratings.csv` and place it in the project directory alongside `movies.csv`.
3. Open `karthik_movie_recommendation.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
4. Update the file paths for `movies.csv` and `ratings.csv` to match your environment (the notebook was originally built for Google Colab with Drive-mounted paths).
5. Run all cells.

### Example Usage

```python
get_movie_recommendation('Iron Man')
```

This returns the top movies most similar to "Iron Man" based on collaborative filtering over user rating patterns.

## Future Improvements

- Add content-based filtering (genres, cast, description) as a complementary approach
- Filter out movies/users with very few ratings to reduce noise
- Wrap the model in a simple web app (Flask/Streamlit) for interactive use
- Add evaluation metrics (precision@k, RMSE) to benchmark recommendation quality

## Author

**G. Karthik Reddy**
GitHub: [@karthik-0211](https://github.com/karthik-0211)
