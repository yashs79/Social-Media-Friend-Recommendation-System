#!/usr/bin/env python3
"""
Demo script for the Hybrid Social Media Friend Recommendation System
"""

import pandas as pd
import matplotlib.pyplot as plt
from recommendation_system import HybridRecommendationSystem
import argparse

def main():
    parser = argparse.ArgumentParser(description='Hybrid Social Media Friend Recommendation System')
    parser.add_argument('--user_id', type=int, default=1, help='User ID to get recommendations for')
    parser.add_argument('--top_n', type=int, default=5, help='Number of recommendations to show')
    parser.add_argument('--bfs_weight', type=float, default=0.3, help='Weight for BFS recommendations')
    parser.add_argument('--dijkstra_weight', type=float, default=0.3, help='Weight for Dijkstra recommendations')
    parser.add_argument('--interest_weight', type=float, default=0.4, help='Weight for interest similarity recommendations')
    parser.add_argument('--visualize', action='store_true', help='Visualize the network')
    args = parser.parse_args()

    # Initialize and load the recommendation system
    print("Initializing recommendation system...")
    recommender = HybridRecommendationSystem(
        user_profiles_path="user_profiles.csv",
        user_relationships_path="user_relationships.csv"
    )
    
    recommender.load_data()
    recommender.build_graph()
    recommender.calculate_interest_similarity()
    
    # Display user information
    user_details = recommender.get_user_details(args.user_id)
    if not user_details:
        print(f"Error: User {args.user_id} not found!")
        return
        
    print("\n" + "="*50)
    print(f"User Profile: {args.user_id}")
    print("="*50)
    print(f"Age: {user_details.get('age')}")
    print(f"Location: {user_details.get('location')}")
    print(f"Interests: {user_details.get('interests')}")
    
    # Get and display different types of recommendations
    print("\n" + "="*50)
    print("Recommendation Methods Comparison")
    print("="*50)
    
    # BFS recommendations
    bfs_recs = recommender.get_bfs_recommendations(args.user_id)
    print(f"\nBFS Recommendations (Friends of Friends):")
    for i, (user_id, depth) in enumerate(bfs_recs[:args.top_n], 1):
        user_details = recommender.get_user_details(user_id)
        print(f"{i}. User {user_id} (Depth: {depth})")
        print(f"   Interests: {user_details.get('interests')}")
    
    # Dijkstra recommendations
    dijkstra_recs = recommender.get_dijkstra_recommendations(args.user_id)
    print(f"\nDijkstra Recommendations (Shortest Path):")
    for i, (user_id, distance) in enumerate(dijkstra_recs[:args.top_n], 1):
        user_details = recommender.get_user_details(user_id)
        print(f"{i}. User {user_id} (Distance: {distance})")
        print(f"   Interests: {user_details.get('interests')}")
    
    # Interest similarity recommendations
    interest_recs = recommender.get_interest_recommendations(args.user_id, top_n=args.top_n)
    print(f"\nInterest Similarity Recommendations:")
    for i, (user_id, score) in enumerate(interest_recs[:args.top_n], 1):
        user_details = recommender.get_user_details(user_id)
        print(f"{i}. User {user_id} (Similarity: {score:.4f})")
        print(f"   Interests: {user_details.get('interests')}")
    
    # Hybrid recommendations
    print("\n" + "="*50)
    print(f"Hybrid Recommendations (Weights: BFS={args.bfs_weight}, Dijkstra={args.dijkstra_weight}, Interest={args.interest_weight})")
    print("="*50)
    
    hybrid_recs = recommender.get_hybrid_recommendations(
        args.user_id, 
        bfs_weight=args.bfs_weight,
        dijkstra_weight=args.dijkstra_weight,
        interest_weight=args.interest_weight,
        top_n=args.top_n
    )
    
    for i, (user_id, score) in enumerate(hybrid_recs, 1):
        user_details = recommender.get_user_details(user_id)
        print(f"{i}. User {user_id} (Score: {score:.4f})")
        print(f"   Age: {user_details.get('age')}")
        print(f"   Location: {user_details.get('location')}")
        print(f"   Interests: {user_details.get('interests')}")
        print()
    
    # Visualize the network if requested
    if args.visualize:
        print("\nVisualizing network around user...")
        recommender.visualize_network(user_id=args.user_id, depth=2)

if __name__ == "__main__":
    main()
