# Hybrid Social Media Friend Recommendation System

This project implements a hybrid recommendation system for social media friend suggestions, combining graph-based algorithms (BFS and Dijkstra) with machine learning techniques for interest similarity.

## Features

- **BFS (Breadth-First Search)**: Finds friends of friends within a specified depth
- **Dijkstra's Algorithm**: Finds shortest paths between users in the social network
- **Interest Similarity**: Uses TF-IDF and cosine similarity to match users with similar interests
- **Hybrid Approach**: Combines all three methods with customizable weights

## Data Files

The system uses two CSV files:

1. `user_profiles.csv`: Contains user information
   - `user_id`: Unique identifier for each user
   - `age`: User's age
   - `location`: User's location
   - `interests`: Comma-separated list of user interests

2. `user_relationships.csv`: Contains social connections
   - `follower_id`: ID of the user who follows
   - `followed_id`: ID of the user being followed

## Usage

### Basic Usage

```python
from recommendation_system import HybridRecommendationSystem

# Initialize the system
recommender = HybridRecommendationSystem(
    user_profiles_path="user_profiles.csv",
    user_relationships_path="user_relationships.csv"
)

# Load data and build models
recommender.load_data()
recommender.build_graph()
recommender.calculate_interest_similarity()

# Get recommendations for a user
recommendations = recommender.get_hybrid_recommendations(
    user_id=1,
    bfs_weight=0.3,
    dijkstra_weight=0.3,
    interest_weight=0.4,
    top_n=5
)
```

### Demo Script

The included `demo.py` script provides a command-line interface to test the recommendation system:

```bash
# Basic usage
python demo.py --user_id 1

# Customize number of recommendations
python demo.py --user_id 1 --top_n 10

# Adjust recommendation weights
python demo.py --user_id 1 --bfs_weight 0.2 --dijkstra_weight 0.2 --interest_weight 0.6

# Visualize the network
python demo.py --user_id 1 --visualize
```

## Requirements

- Python 3.6+
- pandas
- numpy
- networkx
- scikit-learn
- matplotlib

## Installation

```bash
pip install pandas numpy networkx scikit-learn matplotlib
```

## How It Works

1. **Data Loading**: Loads user profiles and relationship data from CSV files
2. **Graph Building**: Constructs a directed graph representing the social network
3. **Interest Analysis**: Vectorizes user interests using TF-IDF and calculates similarity
4. **Recommendation Generation**:
   - BFS finds friends of friends
   - Dijkstra finds users with shortest path connections
   - Interest similarity finds users with matching interests
   - Hybrid approach combines all methods with weighted scores
5. **Visualization**: Optional network visualization around a specific user
