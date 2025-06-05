import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from recommendation_system import HybridRecommendationSystem
import io
from PIL import Image

# Set page title and layout
st.set_page_config(
    page_title="Social Media Friend Recommendation System",
    layout="wide"
)

# Initialize the recommendation system
@st.cache_resource
def load_recommendation_system():
    recommender = HybridRecommendationSystem(
        user_profiles_path="user_profiles.csv",
        user_relationships_path="user_relationships.csv"
    )
    recommender.load_data()
    recommender.build_graph()
    recommender.calculate_interest_similarity()
    return recommender

# Function to generate network visualization
def get_network_visualization(recommender, user_id, depth=1):
    # Create a subgraph centered around the user
    nodes_to_include = {user_id}
    current_nodes = {user_id}
    
    for _ in range(depth):
        next_nodes = set()
        for node in current_nodes:
            if node in recommender.graph:
                next_nodes.update(recommender.graph.successors(node))
                next_nodes.update(recommender.graph.predecessors(node))
        nodes_to_include.update(next_nodes)
        current_nodes = next_nodes
    
    subgraph = recommender.graph.subgraph(nodes_to_include)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(subgraph, seed=42)
    
    # Draw regular nodes
    nx.draw_networkx_nodes(subgraph, pos, 
                          node_color='lightblue', 
                          node_size=300,
                          ax=ax)
    
    # Highlight the center user
    nx.draw_networkx_nodes(subgraph, pos, 
                          nodelist=[user_id], 
                          node_color='red', 
                          node_size=500,
                          ax=ax)
    
    # Draw edges
    nx.draw_networkx_edges(subgraph, pos, arrows=True, ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(subgraph, pos, ax=ax)
    
    plt.title(f"Social Network around User {user_id}")
    plt.axis('off')
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    
    return img

# Main app
def main():
    st.title("Hybrid Social Media Friend Recommendation System")
    st.markdown("""
    This application demonstrates a hybrid friend recommendation system that combines:
    - **BFS** (Breadth-First Search): Finds friends of friends
    - **Dijkstra's Algorithm**: Finds shortest paths between users
    - **Interest Similarity**: Uses ML to match users with similar interests
    """)
    
    # Load the recommendation system
    with st.spinner("Loading recommendation system..."):
        recommender = load_recommendation_system()
    
    # Sidebar for user selection and parameters
    st.sidebar.header("Settings")
    
    # Get all available user IDs
    all_user_ids = sorted(recommender.user_profiles['user_id'].tolist())
    
    # User selection
    user_id = st.sidebar.selectbox("Select User ID", all_user_ids)
    
    # Get user details
    user_details = recommender.get_user_details(user_id)
    
    # Display user profile
    st.header("User Profile")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("User ID", user_id)
    with col2:
        st.metric("Age", user_details.get('age'))
    with col3:
        st.metric("Location", user_details.get('location'))
    
    st.subheader("Interests")
    interests = user_details.get('interests', '')
    if isinstance(interests, str):
        interests_list = [interest.strip() for interest in interests.split(',') if interest.strip()]
        if interests_list:
            st.write(", ".join(interests_list))
        else:
            st.write("No interests specified")
    else:
        st.write("No interests specified")
    
    # Parameters
    st.sidebar.subheader("Recommendation Parameters")
    top_n = st.sidebar.slider("Number of Recommendations", 1, 20, 5)
    
    st.sidebar.subheader("Algorithm Weights")
    bfs_weight = st.sidebar.slider("BFS Weight", 0.0, 1.0, 0.3, 0.1)
    dijkstra_weight = st.sidebar.slider("Dijkstra Weight", 0.0, 1.0, 0.3, 0.1)
    
    # Ensure weights sum to 1.0
    remaining_weight = 1.0 - bfs_weight - dijkstra_weight
    interest_weight = max(0.0, remaining_weight)
    
    st.sidebar.metric("Interest Similarity Weight", f"{interest_weight:.1f}")
    
    # Visualization depth
    viz_depth = st.sidebar.slider("Network Visualization Depth", 1, 3, 1)
    
    # Generate recommendations
    if st.button("Generate Recommendations"):
        with st.spinner("Generating recommendations..."):
            # Get recommendations from each method
            bfs_recs = recommender.get_bfs_recommendations(user_id)
            dijkstra_recs = recommender.get_dijkstra_recommendations(user_id)
            interest_recs = recommender.get_interest_recommendations(user_id, top_n=top_n)
            
            # Get hybrid recommendations
            hybrid_recs = recommender.get_hybrid_recommendations(
                user_id, 
                bfs_weight=bfs_weight,
                dijkstra_weight=dijkstra_weight,
                interest_weight=interest_weight,
                top_n=top_n
            )
            
            # Display recommendations in tabs
            tab1, tab2, tab3, tab4 = st.tabs(["Hybrid", "BFS", "Dijkstra", "Interest"])
            
            # Hybrid recommendations
            with tab1:
                st.subheader(f"Hybrid Recommendations (Weights: BFS={bfs_weight}, Dijkstra={dijkstra_weight}, Interest={interest_weight})")
                if hybrid_recs:
                    for i, (rec_id, score) in enumerate(hybrid_recs, 1):
                        rec_details = recommender.get_user_details(rec_id)
                        with st.expander(f"{i}. User {rec_id} (Score: {score:.4f})"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Age:** {rec_details.get('age')}")
                                st.write(f"**Location:** {rec_details.get('location')}")
                            with col2:
                                interests = rec_details.get('interests', '')
                                if isinstance(interests, str):
                                    interests_list = [interest.strip() for interest in interests.split(',') if interest.strip()]
                                    if interests_list:
                                        st.write(f"**Interests:** {', '.join(interests_list)}")
                                    else:
                                        st.write("**Interests:** None specified")
                                else:
                                    st.write("**Interests:** None specified")
                else:
                    st.info("No hybrid recommendations found.")
            
            # BFS recommendations
            with tab2:
                st.subheader("BFS Recommendations (Friends of Friends)")
                if bfs_recs:
                    for i, (rec_id, depth) in enumerate(bfs_recs[:top_n], 1):
                        rec_details = recommender.get_user_details(rec_id)
                        with st.expander(f"{i}. User {rec_id} (Depth: {depth})"):
                            st.write(f"**Interests:** {rec_details.get('interests')}")
                else:
                    st.info("No BFS recommendations found.")
            
            # Dijkstra recommendations
            with tab3:
                st.subheader("Dijkstra Recommendations (Shortest Path)")
                if dijkstra_recs:
                    for i, (rec_id, distance) in enumerate(dijkstra_recs[:top_n], 1):
                        rec_details = recommender.get_user_details(rec_id)
                        with st.expander(f"{i}. User {rec_id} (Distance: {distance})"):
                            st.write(f"**Interests:** {rec_details.get('interests')}")
                else:
                    st.info("No Dijkstra recommendations found.")
            
            # Interest similarity recommendations
            with tab4:
                st.subheader("Interest Similarity Recommendations")
                if interest_recs:
                    for i, (rec_id, similarity) in enumerate(interest_recs[:top_n], 1):
                        rec_details = recommender.get_user_details(rec_id)
                        with st.expander(f"{i}. User {rec_id} (Similarity: {similarity:.4f})"):
                            st.write(f"**Interests:** {rec_details.get('interests')}")
                else:
                    st.info("No interest similarity recommendations found.")
            
            # Network visualization
            st.header("Network Visualization")
            with st.spinner("Generating network visualization..."):
                network_img = get_network_visualization(recommender, user_id, depth=viz_depth)
                st.image(network_img, caption=f"Social Network around User {user_id} (Depth: {viz_depth})")

if __name__ == "__main__":
    main()
