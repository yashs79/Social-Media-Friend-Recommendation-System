import pandas as pd
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, deque
import heapq
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Set, Any

class HybridRecommendationSystem:
    def __init__(self, user_profiles_path: str, user_relationships_path: str):
        """
        Initialize the hybrid recommendation system with data paths
        
        Args:
            user_profiles_path: Path to the user profiles CSV
            user_relationships_path: Path to the user relationships CSV
        """
        self.user_profiles_path = user_profiles_path
        self.user_relationships_path = user_relationships_path
        self.user_profiles = None
        self.user_relationships = None
        self.graph = nx.DiGraph()
        self.interest_similarity_matrix = None
        self.user_interests_vector = None
        
    def load_data(self) -> None:
        """Load and process the user profiles and relationships data"""
        print("Loading data...")
        # Load user profiles
        self.user_profiles = pd.read_csv(self.user_profiles_path)
        # Load user relationships
        self.user_relationships = pd.read_csv(self.user_relationships_path)
        
        print(f"Loaded {len(self.user_profiles)} user profiles")
        print(f"Loaded {len(self.user_relationships)} user relationships")
        
        # Fill NaN interests with empty string
        self.user_profiles['interests'] = self.user_profiles['interests'].fillna('')
        
    def build_graph(self) -> None:
        """Build a directed graph from user relationships"""
        print("Building social network graph...")
        
        # Add all users as nodes
        for user_id in self.user_profiles['user_id']:
            self.graph.add_node(user_id)
        
        # Add relationships as edges
        for _, row in self.user_relationships.iterrows():
            self.graph.add_edge(row['follower_id'], row['followed_id'])
            
        print(f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        
    def calculate_interest_similarity(self) -> None:
        """Calculate interest similarity between users using TF-IDF and cosine similarity"""
        print("Calculating interest similarity...")
        
        # Create a user-interests dictionary
        user_interests = self.user_profiles.set_index('user_id')['interests'].to_dict()
        
        # Convert interests to strings for TF-IDF
        user_ids = list(user_interests.keys())
        interest_texts = [str(user_interests[uid]) for uid in user_ids]
        
        # Use TF-IDF to vectorize interests
        vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'[^,\s]+')
        self.user_interests_vector = vectorizer.fit_transform(interest_texts)
        
        # Calculate cosine similarity
        self.interest_similarity_matrix = cosine_similarity(self.user_interests_vector)
        
        print("Interest similarity matrix calculated")
        
    def get_bfs_recommendations(self, user_id: int, max_depth: int = 2) -> List[Tuple[int, int]]:
        """
        Get friend recommendations using BFS to find friends of friends
        
        Args:
            user_id: The user to get recommendations for
            max_depth: Maximum depth for BFS search (default: 2 for friends of friends)
            
        Returns:
            List of (recommended_user_id, depth) tuples
        """
        if user_id not in self.graph:
            return []
        
        visited = {user_id}
        queue = deque([(user_id, 0)])  # (node, depth)
        recommendations = []
        
        # Get current friends to exclude them from recommendations
        current_friends = set(self.graph.successors(user_id))
        visited.update(current_friends)
        
        while queue:
            node, depth = queue.popleft()
            
            if depth >= max_depth:
                continue
                
            for neighbor in self.graph.successors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
                    if neighbor != user_id and neighbor not in current_friends:
                        recommendations.append((neighbor, depth + 1))
        
        return recommendations
    
    def get_dijkstra_recommendations(self, user_id: int, max_distance: int = 3) -> List[Tuple[int, float]]:
        """
        Get friend recommendations using Dijkstra's algorithm to find shortest paths
        
        Args:
            user_id: The user to get recommendations for
            max_distance: Maximum distance to consider for recommendations
            
        Returns:
            List of (recommended_user_id, distance) tuples
        """
        if user_id not in self.graph:
            return []
        
        # Get current friends to exclude them from recommendations
        current_friends = set(self.graph.successors(user_id))
        
        # Initialize distances
        distances = {node: float('infinity') for node in self.graph.nodes()}
        distances[user_id] = 0
        priority_queue = [(0, user_id)]
        
        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)
            
            # If we've processed this node already at a shorter distance, skip
            if current_distance > distances[current_node]:
                continue
                
            # Process neighbors
            for neighbor in self.graph.successors(current_node):
                distance = current_distance + 1
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(priority_queue, (distance, neighbor))
        
        # Filter recommendations
        recommendations = []
        for node, distance in distances.items():
            if (node != user_id and 
                node not in current_friends and 
                distance <= max_distance and 
                distance > 0):
                recommendations.append((node, distance))
                
        return recommendations
    
    def get_interest_recommendations(self, user_id: int, top_n: int = 10) -> List[Tuple[int, float]]:
        """
        Get friend recommendations based on interest similarity
        
        Args:
            user_id: The user to get recommendations for
            top_n: Number of top recommendations to return
            
        Returns:
            List of (recommended_user_id, similarity_score) tuples
        """
        # Find the index of the user in our matrix
        user_index = self.user_profiles[self.user_profiles['user_id'] == user_id].index
        if len(user_index) == 0:
            return []
        
        user_index = user_index[0]
        
        # Get current friends to exclude them from recommendations
        current_friends = set(self.graph.successors(user_id))
        
        # Get similarity scores for all users
        similarity_scores = []
        for i, score in enumerate(self.interest_similarity_matrix[user_index]):
            if i != user_index:
                other_user_id = self.user_profiles.iloc[i]['user_id']
                if other_user_id not in current_friends and other_user_id != user_id:
                    similarity_scores.append((other_user_id, score))
        
        # Sort by similarity score and take top N
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        return similarity_scores[:top_n]
    
    def get_hybrid_recommendations(self, user_id: int, bfs_weight: float = 0.3, 
                                  dijkstra_weight: float = 0.3, interest_weight: float = 0.4,
                                  top_n: int = 10) -> List[Tuple[int, float]]:
        """
        Get hybrid recommendations combining BFS, Dijkstra, and interest similarity
        
        Args:
            user_id: The user to get recommendations for
            bfs_weight: Weight for BFS recommendations
            dijkstra_weight: Weight for Dijkstra recommendations
            interest_weight: Weight for interest similarity recommendations
            top_n: Number of top recommendations to return
            
        Returns:
            List of (recommended_user_id, score) tuples
        """
        # Get recommendations from each method
        bfs_recs = self.get_bfs_recommendations(user_id)
        dijkstra_recs = self.get_dijkstra_recommendations(user_id)
        interest_recs = self.get_interest_recommendations(user_id, top_n=len(self.user_profiles))
        
        # Normalize scores
        max_bfs_depth = max([depth for _, depth in bfs_recs]) if bfs_recs else 1
        max_dijkstra_dist = max([dist for _, dist in dijkstra_recs]) if dijkstra_recs else 1
        
        # Create score dictionaries
        bfs_scores = {uid: 1 - (depth / (max_bfs_depth + 1)) for uid, depth in bfs_recs}
        dijkstra_scores = {uid: 1 - (dist / (max_dijkstra_dist + 1)) for uid, dist in dijkstra_recs}
        interest_scores = {uid: score for uid, score in interest_recs}
        
        # Combine scores
        combined_scores = defaultdict(float)
        all_users = set(list(bfs_scores.keys()) + list(dijkstra_scores.keys()) + list(interest_scores.keys()))
        
        for uid in all_users:
            bfs_score = bfs_scores.get(uid, 0)
            dijkstra_score = dijkstra_scores.get(uid, 0)
            interest_score = interest_scores.get(uid, 0)
            
            combined_scores[uid] = (
                bfs_weight * bfs_score +
                dijkstra_weight * dijkstra_score +
                interest_weight * interest_score
            )
        
        # Sort and return top N recommendations
        recommendations = [(uid, score) for uid, score in combined_scores.items()]
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:top_n]
    
    def get_user_details(self, user_id: int) -> Dict[str, Any]:
        """Get details for a specific user"""
        user_data = self.user_profiles[self.user_profiles['user_id'] == user_id]
        if len(user_data) == 0:
            return {}
        
        return user_data.iloc[0].to_dict()
    
    def visualize_network(self, user_id: int = None, depth: int = 1) -> None:
        """
        Visualize the social network graph
        
        Args:
            user_id: Center user for the visualization (optional)
            depth: Depth of connections to visualize from center user
        """
        if user_id is not None:
            # Create a subgraph centered around the user
            nodes_to_include = {user_id}
            current_nodes = {user_id}
            
            for _ in range(depth):
                next_nodes = set()
                for node in current_nodes:
                    next_nodes.update(self.graph.successors(node))
                    next_nodes.update(self.graph.predecessors(node))
                nodes_to_include.update(next_nodes)
                current_nodes = next_nodes
            
            subgraph = self.graph.subgraph(nodes_to_include)
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(subgraph)
            
            # Draw regular nodes
            nx.draw_networkx_nodes(subgraph, pos, 
                                  node_color='lightblue', 
                                  node_size=300)
            
            # Highlight the center user
            nx.draw_networkx_nodes(subgraph, pos, 
                                  nodelist=[user_id], 
                                  node_color='red', 
                                  node_size=500)
            
            # Draw edges
            nx.draw_networkx_edges(subgraph, pos, arrows=True)
            
            # Draw labels
            nx.draw_networkx_labels(subgraph, pos)
            
            plt.title(f"Social Network around User {user_id}")
            plt.axis('off')
            plt.show()
        else:
            # If the graph is too large, sample it
            if self.graph.number_of_nodes() > 100:
                sampled_nodes = np.random.choice(list(self.graph.nodes()), 
                                               size=100, 
                                               replace=False)
                subgraph = self.graph.subgraph(sampled_nodes)
            else:
                subgraph = self.graph
            
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(subgraph)
            nx.draw(subgraph, pos, with_labels=True, node_color='lightblue', 
                   node_size=300, arrows=True)
            plt.title("Social Network Graph Sample")
            plt.axis('off')
            plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize the recommendation system
    recommender = HybridRecommendationSystem(
        user_profiles_path="user_profiles.csv",
        user_relationships_path="user_relationships.csv"
    )
    
    # Load data and build the model
    recommender.load_data()
    recommender.build_graph()
    recommender.calculate_interest_similarity()
    
    # Get recommendations for a sample user
    sample_user_id = 1
    print(f"\nGetting recommendations for user {sample_user_id}:")
    
    # Get hybrid recommendations
    recommendations = recommender.get_hybrid_recommendations(sample_user_id, top_n=5)
    
    # Display recommendations with user details
    print("\nTop 5 Recommendations:")
    for i, (user_id, score) in enumerate(recommendations, 1):
        user_details = recommender.get_user_details(user_id)
        print(f"{i}. User {user_id} (Score: {score:.4f})")
        print(f"   Age: {user_details.get('age')}")
        print(f"   Location: {user_details.get('location')}")
        print(f"   Interests: {user_details.get('interests')}")
        print()
    
    # Visualize the network around the sample user
    # Uncomment to visualize (requires matplotlib)
    # recommender.visualize_network(user_id=sample_user_id, depth=2)
