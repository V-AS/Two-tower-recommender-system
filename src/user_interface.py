"""
Terminal-based User Interface for Book Recommendation System
Run this script to get book recommendations through the command line.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import argparse
from colorama import init, Fore, Style

# Initialize colorama for cross-platform colored terminal output
init()

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import modules from your existing system
from modules.neural_network import create_user_tower
from modules.embedding_generation import EmbeddingGenerator
from modules.recommendation import Recommender
from modules.ann_search import ANNSearch
from hardware.system_interface import load_model
from modules.data_processing import DataProcessor

# Configuration
OUTPUT_DIR = "output"
DATA_PATH = "data/processed/recommender_data.csv"
NUM_RECOMMENDATIONS = 10

def initialize_system():
    """Initialize the recommendation system components."""
    print(f"{Fore.BLUE}Initializing recommendation system...{Style.RESET_ALL}")
    
    # Load data processor
    data_processor = DataProcessor()
    
    # Load models
    user_model = load_model(os.path.join(OUTPUT_DIR, "user_model.pth"))
    item_model = load_model(os.path.join(OUTPUT_DIR, "item_model.pth"))
    
    # Load ANN index
    ann_search = ANNSearch()
    ann_index = ann_search.load_index(os.path.join(OUTPUT_DIR, "ann_index"))
    
    # Load book mapping
    book_mapping = np.load(os.path.join(OUTPUT_DIR, "book_mapping.npy"), allow_pickle=True).item()
    
    # Initialize embedding generator
    embedding_generator = EmbeddingGenerator()
    embedding_generator.initialize(user_model, item_model)
    
    # Initialize recommender
    recommender = Recommender()
    recommender.initialize(ann_index, embedding_generator, book_mapping)
    
    # Load data for preprocessing
    try:
        data = pd.read_csv(DATA_PATH)
        
        # Get state and country mappings from data
        unique_states = data['State'].dropna().unique()
        state_map = {state: idx for idx, state in enumerate(unique_states)}
        
        unique_countries = data['Country'].dropna().unique()
        country_map = {country: idx for idx, country in enumerate(unique_countries)}
        
        # Get age range for normalization
        min_age = data['Age'].min()
        max_age = data['Age'].max()
        
        print(f"Loaded {len(state_map)} states and {len(country_map)} countries")
    except Exception as e:
        print(f"{Fore.RED}Error loading data for mappings: {e}{Style.RESET_ALL}")
        # Use fallback mappings
        state_map = {"Unknown": 0}
        country_map = {"Unknown": 0}
        min_age = 0
        max_age = 100
    
    print(f"{Fore.GREEN}System initialized successfully{Style.RESET_ALL}")
    
    return {
        'user_model': user_model,
        'item_model': item_model, 
        'recommender': recommender,
        'embedding_generator': embedding_generator,
        'state_map': state_map,
        'country_map': country_map,
        'min_age': min_age,
        'max_age': max_age
    }

def create_user_features(age, state, country, state_map, country_map, min_age, max_age):
    """
    Create user feature vector from input.
    
    Args:
        age (float): User age
        state (str): User state
        country (str): User country
        
    Returns:
        numpy.ndarray: User feature vector
    """
    # Normalize age to [0, 1] range
    age_normalized = (age - min_age) / (max_age - min_age) if max_age > min_age else 0.5
    
    # Encode state and country
    state_encoded = state_map.get(state, 0)
    country_encoded = country_map.get(country, 0)
    
    # Create feature vector
    user_features = np.array([age_normalized, state_encoded, country_encoded])
    
    return user_features

def display_recommendations(recommendations):
    """Format and display recommendations in the terminal."""
    print("\n" + "=" * 80)
    print(f"{Fore.CYAN}YOUR BOOK RECOMMENDATIONS{Style.RESET_ALL}")
    print("=" * 80)
    
    for i, rec in enumerate(recommendations, 1):
        title = rec.get('title', 'Unknown Title')
        author = rec.get('author', 'Unknown Author')
        year = rec.get('year', 'Unknown Year')
        publisher = rec.get('publisher', 'Unknown Publisher')
        score = rec.get('score', 0) * 100
        
        print(f"\n{Fore.GREEN}#{i}: {Fore.YELLOW}{title}{Style.RESET_ALL}")
        print(f"   Author: {Fore.CYAN}{author}{Style.RESET_ALL}")
        print(f"   Published: {year} by {publisher}")
        print(f"   {Fore.MAGENTA}Match Score: {score:.1f}%{Style.RESET_ALL}")
    
    print("\n" + "=" * 80)

def interactive_mode(system):
    """Run the recommendation system in interactive mode."""
    recommender = system['recommender']
    state_map = system['state_map']
    country_map = system['country_map']
    min_age = system['min_age']
    max_age = system['max_age']
    
    print("\n" + "=" * 80)
    print(f"{Fore.CYAN}BOOK RECOMMENDATION SYSTEM{Style.RESET_ALL}")
    print("=" * 80)
    
    while True:
        print("\nPlease enter your information to get personalized book recommendations:")
        
        # Get age input
        while True:
            try:
                age = float(input(f"Your age ({int(min_age)}-{int(max_age)}): "))
                if age < min_age or age > max_age:
                    print(f"{Fore.YELLOW}Warning: Age outside the expected range. Using closest valid value.{Style.RESET_ALL}")
                    age = max(min_age, min(age, max_age))
                break
            except ValueError:
                print(f"{Fore.RED}Please enter a valid number for age.{Style.RESET_ALL}")
        

        states_list = sorted(state_map.keys())
        # for i in range(0, len(states_list), 5):
        #     chunk = states_list[i:i+5]
        #     print("  " + ", ".join(chunk))
        
        # Get state input
        state = input("\nYour state/province (press Enter for Unknown): ").strip()
        if not state:
            state = "Unknown"
        elif state not in state_map:
            closest = [s for s in states_list if state.lower() in s.lower()]
            if closest:
                print(f"{Fore.YELLOW}State not found. Did you mean one of these? {', '.join(closest[:3])}{Style.RESET_ALL}")
                state = input("Your state/province (press Enter for Unknown): ").strip()
                if not state:
                    state = "Unknown"
            else:
                print(f"{Fore.YELLOW}State not found. Using 'Unknown'.{Style.RESET_ALL}")
                state = "Unknown"
        
        countries_list = sorted(country_map.keys())
        # for i in range(0, len(countries_list), 5):
        #     chunk = countries_list[i:i+5]
        #     print("  " + ", ".join(chunk))
        
        # Get country input
        country = input("\nYour country (press Enter for Unknown): ").strip()
        if not country:
            country = "Unknown"
        elif country not in country_map:
            closest = [c for c in countries_list if country.lower() in c.lower()]
            if closest:
                print(f"{Fore.YELLOW}Country not found. Did you mean one of these? {', '.join(closest[:3])}{Style.RESET_ALL}")
                country = input("Your country (press Enter for Unknown): ").strip()
                if not country:
                    country = "Unknown"
            else:
                print(f"{Fore.YELLOW}Country not found. Using 'Unknown'.{Style.RESET_ALL}")
                country = "Unknown"
        
        # Create user features
        print(f"\n{Fore.BLUE}Generating recommendations for: Age {age}, {state}, {country}{Style.RESET_ALL}")
        print("Please wait...")
        
        user_features = create_user_features(age, state, country, state_map, country_map, min_age, max_age)
        
        # Generate recommendations
        recommendations = recommender.get_recommendations(
            user_features, 
            num_results=NUM_RECOMMENDATIONS
        )
        
        # Display recommendations
        display_recommendations(recommendations)
        
        # Ask if user wants to continue
        continue_response = input("\nWould you like to get more recommendations? (y/n): ").strip().lower()
        if continue_response != 'y':
            print(f"\n{Fore.GREEN}Thank you for using the Book Recommendation System!{Style.RESET_ALL}")
            break

def main():
    """Main function to start the application."""
    parser = argparse.ArgumentParser(description="Book Recommendation System")
    
    parser.add_argument("--age", type=float, help="Your age")
    parser.add_argument("--state", type=str, help="Your state or province")
    parser.add_argument("--country", type=str, help="Your country")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Initialize the system
    system = initialize_system()
    
    if args.interactive or (args.age is None and args.state is None and args.country is None):
        # Run in interactive mode
        interactive_mode(system)
    else:
        # Use command line arguments
        age = args.age if args.age is not None else 30
        state = args.state if args.state is not None else "Unknown"
        country = args.country if args.country is not None else "Unknown"
        
        # Create user features
        user_features = create_user_features(
            age, state, country, 
            system['state_map'], system['country_map'], 
            system['min_age'], system['max_age']
        )
        
        # Generate recommendations
        print(f"{Fore.BLUE}Generating recommendations for: Age {age}, {state}, {country}{Style.RESET_ALL}")
        recommendations = system['recommender'].get_recommendations(
            user_features, 
            num_results=NUM_RECOMMENDATIONS,
            diversity_weight=0.2
        )
        
        # Display recommendations
        display_recommendations(recommendations)

if __name__ == "__main__":
    main()