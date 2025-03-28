"""
Enhanced Terminal-based User Interface for Book Recommendation System
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

# Import modules from your system
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
    
    # Load the actual dataset to get state/country statistics
    try:
        data = pd.read_csv(DATA_PATH)
        print(f"Loaded dataset with {len(data)} records")
        
        # Check available columns
        print(f"Available columns: {data.columns.tolist()}")
        
        # Process data to get frequencies
        processed_data = data_processor.preprocess_data(data)
        
        # Get state and country frequencies
        state_counts = processed_data['State'].value_counts()
        country_counts = processed_data['Country'].value_counts()
        
        # Create frequency mappings
        state_freq = {state.strip().lower(): count/len(processed_data) for state, count in state_counts.items()}
        country_freq = {country.strip().lower(): count/len(processed_data) for country, count in country_counts.items()}
        
        # Get age range for normalization
        min_age = processed_data['Age'].min()
        max_age = processed_data['Age'].max()
        
        # Define age groups
        age_bins = [0, 18, 25, 35, 50, 100]
        
        print(f"Loaded {len(state_counts)} states and {len(country_counts)} countries")
        
    except Exception as e:
        print(f"{Fore.RED}Error loading data for mappings: {e}{Style.RESET_ALL}")
        # Use fallback mappings
        state_freq = {"Unknown": 0.5}
        country_freq = {"Unknown": 0.5}
        state_counts = pd.Series([100], index=["Unknown"])
        country_counts = pd.Series([100], index=["Unknown"])
        min_age = 0
        max_age = 100
        age_bins = [0, 18, 25, 35, 50, 100]
    
    # Load models
    try:
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
    except Exception as e:
        print(f"{Fore.RED}Error loading models: {e}{Style.RESET_ALL}")
        raise
    
    print(f"{Fore.GREEN}System initialized successfully{Style.RESET_ALL}")
    
    return {
        'user_model': user_model,
        'item_model': item_model, 
        'recommender': recommender,
        'embedding_generator': embedding_generator,
        'state_freq': state_freq,
        'country_freq': country_freq,
        'state_counts': state_counts,
        'country_counts': country_counts,
        'min_age': min_age,
        'max_age': max_age,
        'age_bins': age_bins
    }

def create_user_features(age, state, country, system):
    """
    Create user feature vector matching our model expectations.
    
    Args:
        age (float): User age
        state (str): User state
        country (str): User country
        system (dict): System information including mappings
        
    Returns:
        numpy.ndarray: User feature vector
    """
    # Get required information from system
    state_freq = system['state_freq']
    country_freq = system['country_freq']
    min_age = system['min_age']
    max_age = system['max_age']
    age_bins = system['age_bins']
    
    # 1. Normalize age to [0, 1] range
    age_normalized = (age - min_age) / (max_age - min_age) if max_age > min_age else 0.5
    
    # 2. Create age group (0-4)
    age_group = 0
    for i in range(len(age_bins)-1):
        if age_bins[i] <= age < age_bins[i+1]:
            age_group = i
            break
    
    # 3. Get state and country frequencies
    state_frequency = state_freq.get(state.lower(), 0.0)
    country_frequency = country_freq.get(country.lower(), 0.0)
    
    # Create feature vector - matches our data processor output
    user_features = np.array([
        age_normalized,    # Age-Normalized
        float(age_group),  # Age-Group
        state_frequency,   # State-Frequency
        country_frequency  # Country-Frequency
    ])
    
    # Debug output
    print(f"Created user features: {user_features}")
    
    return user_features

def display_top_regions(state_counts, country_counts, top_n=10):
    """Display the top states and countries in the dataset."""
    print("\n" + "=" * 80)
    print(f"{Fore.CYAN}TOP {top_n} STATES/PROVINCES IN THE DATASET{Style.RESET_ALL}")
    print("-" * 80)
    
    # Display top states
    for i, (state, count) in enumerate(state_counts.head(top_n).items(), 1):
        percent = count / state_counts.sum() * 100
        print(f"{i}. {Fore.GREEN}{state}{Style.RESET_ALL}: {count} users ({percent:.1f}%)")
    
    print("\n" + "=" * 80)
    print(f"{Fore.CYAN}TOP {top_n} COUNTRIES IN THE DATASET{Style.RESET_ALL}")
    print("-" * 80)
    
    # Display top countries
    for i, (country, count) in enumerate(country_counts.head(top_n).items(), 1):
        percent = count / country_counts.sum() * 100
        print(f"{i}. {Fore.GREEN}{country}{Style.RESET_ALL}: {count} users ({percent:.1f}%)")
    
    print("=" * 80)

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
        score = rec.get('score', 0)
        
        print(f"\n{Fore.GREEN}#{i}: {Fore.YELLOW}{title}{Style.RESET_ALL}")
        print(f"   Author: {Fore.CYAN}{author}{Style.RESET_ALL}")
        print(f"   Published: {year} by {publisher}")
        print(f"   {Fore.MAGENTA}Estimated Rating: {(score+1)*5:.1f}%{Style.RESET_ALL}")
    
    print("\n" + "=" * 80)

def interactive_mode(system):
    """Run the recommendation system in interactive mode."""
    recommender = system['recommender']
    state_counts = system['state_counts']
    country_counts = system['country_counts']
    min_age = system['min_age']
    max_age = system['max_age']
    
    # Display top regions first
    display_top_regions(state_counts, country_counts)
    
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
        
        # Get state input - show top options
        top_states = state_counts.head(10).index.tolist()
        print(f"\nTop 10 states/provinces (for reference): {', '.join(top_states)}")
        state = input("Your state/province (press Enter for Unknown): ").strip()
        if not state:
            state = "Unknown"
        
        # Get country input - show top options
        top_countries = country_counts.head(10).index.tolist()
        print(f"\nTop 10 countries (for reference): {', '.join(top_countries)}")
        country = input("Your country (press Enter for Unknown): ").strip()
        if not country:
            country = "Unknown"
        
        # Create user features
        print(f"\n{Fore.BLUE}Generating recommendations for: Age {age}, {state}, {country}{Style.RESET_ALL}")
        print("Please wait...")
        
        user_features = create_user_features(age, state, country, system)
        
        # Generate recommendations
        try:
            recommendations = recommender.get_recommendations(
                user_features, 
                num_results=NUM_RECOMMENDATIONS
            )
            
            # Display recommendations
            display_recommendations(recommendations)
        except Exception as e:
            print(f"{Fore.RED}Error generating recommendations: {e}{Style.RESET_ALL}")
        
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
        
        # Display top regions
        display_top_regions(system['state_counts'], system['country_counts'])
        
        # Create user features
        user_features = create_user_features(age, state, country, system)
        
        # Generate recommendations
        print(f"{Fore.BLUE}Generating recommendations for: Age {age}, {state}, {country}{Style.RESET_ALL}")
        try:
            recommendations = system['recommender'].get_recommendations(
                user_features, 
                num_results=NUM_RECOMMENDATIONS
            )
            
            # Display recommendations
            display_recommendations(recommendations)
        except Exception as e:
            print(f"{Fore.RED}Error generating recommendations: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()