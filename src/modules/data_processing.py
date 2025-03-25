"""
Data Processing Module for loading and preprocessing data.
Handles dataset validation, preprocessing, and splitting.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class DataProcessor:
    def __init__(self):
        """Initialize the DataProcessor."""
        self.users_df = None
        self.books_df = None
        self.ratings_df = None
        self.user_encoder = LabelEncoder()
        self.isbn_encoder = LabelEncoder()
        self.location_encoder = None  # Will be initialized during preprocessing
        
    def load_data(self, users_path, books_path, ratings_path):
        """
        Load data from CSV files.
        
        Args:
            users_path (str): Path to Users.csv
            books_path (str): Path to Books.csv
            ratings_path (str): Path to Ratings.csv
            
        Returns:
            DataSet: A dataset object containing the loaded data
            
        Raises:
            IOError: If files cannot be read
            FormatError: If file format is invalid
        """
        try:
            self.users_df = pd.read_csv(users_path)
            self.books_df = pd.read_csv(books_path)
            self.ratings_df = pd.read_csv(ratings_path)
            
            # Create a dataset object to return
            dataset = {
                'users': self.users_df,
                'books': self.books_df,
                'ratings': self.ratings_df
            }
            
            return dataset
        except pd.errors.EmptyDataError:
            raise ValueError("One of the CSV files is empty")
        except pd.errors.ParserError:
            raise ValueError("Error parsing CSV format")
        except Exception as e:
            raise IOError(f"Failed to load data: {str(e)}")
    
    def validate_data(self, dataset):
        """
        Validate the loaded data.
        
        Args:
            dataset: The dataset to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            users_df = dataset['users']
            books_df = dataset['books']
            ratings_df = dataset['ratings']
            
            # Check for required columns
            required_user_cols = ['User-ID', 'Location', 'Age']
            required_book_cols = ['ISBN', 'Book-Title', 'Book-Author']
            required_rating_cols = ['User-ID', 'ISBN', 'Book-Rating']
            
            if not all(col in users_df.columns for col in required_user_cols):
                return False
            if not all(col in books_df.columns for col in required_book_cols):
                return False
            if not all(col in ratings_df.columns for col in required_rating_cols):
                return False
                
            # Check for matching IDs between ratings and users/books
            if not set(ratings_df['User-ID']).issubset(set(users_df['User-ID'])):
                return False
            if not set(ratings_df['ISBN']).issubset(set(books_df['ISBN'])):
                return False
                
            return True
        except Exception:
            return False
    
    def preprocess_data(self, dataset):
        """
        Preprocess the loaded data.
        
        Args:
            dataset: The dataset to preprocess
            
        Returns:
            Processed dataset ready for model training
        """
        users_df = dataset['users'].copy()
        books_df = dataset['books'].copy()
        ratings_df = dataset['ratings'].copy()
        
        # Encode categorical variables
        users_df['User-ID-Encoded'] = self.user_encoder.fit_transform(users_df['User-ID'])
        books_df['ISBN-Encoded'] = self.isbn_encoder.fit_transform(books_df['ISBN'])
        
        # Map encoded IDs to ratings
        ratings_df = ratings_df.merge(users_df[['User-ID', 'User-ID-Encoded']], on='User-ID')
        ratings_df = ratings_df.merge(books_df[['ISBN', 'ISBN-Encoded']], on='ISBN')
        
        # Normalize ratings to [0, 1] range
        ratings_df['Normalized-Rating'] = ratings_df['Book-Rating'] / 10.0
        
        # Create user features
        users_df['Age'] = users_df['Age'].fillna(users_df['Age'].median())
        users_df['Age-Normalized'] = (users_df['Age'] - users_df['Age'].min()) / (users_df['Age'].max() - users_df['Age'].min())
        
        # Clean locations
        users_df['Location'] = users_df['Location'].astype(str).apply(lambda x: x.strip())
        
        # Limit to top N locations to prevent feature explosion
        location_counts = users_df['Location'].value_counts()
        top_locations = location_counts[location_counts >= 5].index.tolist()
        
        # Add 'Other' category for rare locations
        users_df['Location_Category'] = users_df['Location'].apply(
            lambda x: x if x in top_locations else 'Other'
        )
        
        # Encode locations
        self.location_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        location_encoded = self.location_encoder.fit_transform(users_df[['Location_Category']])
        
        # Create DataFrame with encoded location features
        location_feature_names = [f'location_{i}' for i in range(location_encoded.shape[1])]
        location_features_df = pd.DataFrame(
            location_encoded, 
            columns=location_feature_names,
            index=users_df.index
        )
        
        # Concatenate with original dataframe
        users_df = pd.concat([users_df, location_features_df], axis=1)
        
        # Create book features
        # Extract publication year and normalize
        if 'Year-Of-Publication' in books_df.columns:
            books_df['Year-Of-Publication'] = pd.to_numeric(books_df['Year-Of-Publication'], errors='coerce')
            books_df['Year-Of-Publication'] = books_df['Year-Of-Publication'].fillna(books_df['Year-Of-Publication'].median())
            books_df['Year-Normalized'] = (books_df['Year-Of-Publication'] - books_df['Year-Of-Publication'].min()) / \
                                          (books_df['Year-Of-Publication'].max() - books_df['Year-Of-Publication'].min() + 1e-10)
        
        # Create author embeddings based on frequency
        author_counts = books_df['Book-Author'].value_counts()
        books_df['Author-Popularity'] = books_df['Book-Author'].map(author_counts) / author_counts.max()
        
        processed_dataset = {
            'users': users_df,
            'books': books_df,
            'ratings': ratings_df,
            'user_encoder': self.user_encoder,
            'isbn_encoder': self.isbn_encoder,
            'location_encoder': self.location_encoder,
            'top_locations': top_locations
        }
        
        return processed_dataset
    
    def process_new_user(self, user_data):
        """
        Process a new user's data for prediction.
        
        Args:
            user_data (dict): Dictionary with user information (Location, Age)
            
        Returns:
            array: Processed user features ready for embedding generation
        """
        if self.location_encoder is None:
            raise ValueError("DataProcessor must be initialized with preprocess_data first")
        
        # Create DataFrame with user data
        user_df = pd.DataFrame([user_data])
        
        # Normalize age using same logic as in preprocess_data
        user_df['Age-Normalized'] = (user_df['Age'] - self.users_df['Age'].min()) / \
                                    (self.users_df['Age'].max() - self.users_df['Age'].min())
        
        # Process location
        location = user_data['Location'].strip()
        
        # Check if location is in top locations, otherwise mark as 'Other'
        if hasattr(self, 'top_locations'):
            location_category = location if location in self.top_locations else 'Other'
        else:
            location_category = 'Other'
            
        # One-hot encode location
        location_encoded = self.location_encoder.transform([[location_category]])
        
        # Combine features (age + location)
        user_features = np.concatenate([
            user_df[['Age-Normalized']].values,
            location_encoded
        ], axis=1)
        
        return user_features
    
    def split_data(self, dataset, train_ratio=0.8):
        """
        Split the dataset into training and testing sets.
        
        Args:
            dataset: The dataset to split
            train_ratio (float): Ratio of training data
            
        Returns:
            train_data, test_data: The split datasets
            
        Raises:
            ValueError: If train_ratio is not in (0, 1)
        """
        if train_ratio <= 0 or train_ratio >= 1:
            raise ValueError("train_ratio must be in (0, 1)")
        
        ratings_df = dataset['ratings']
        
        # Split ratings
        train_ratings, test_ratings = train_test_split(
            ratings_df, train_size=train_ratio, random_state=42
        )
        
        # Create training and testing datasets
        train_data = {
            'users': dataset['users'],
            'books': dataset['books'],
            'ratings': train_ratings,
            'user_encoder': dataset['user_encoder'],
            'isbn_encoder': dataset['isbn_encoder'],
            'location_encoder': dataset.get('location_encoder'),
            'top_locations': dataset.get('top_locations')
        }
        
        test_data = {
            'users': dataset['users'],
            'books': dataset['books'],
            'ratings': test_ratings,
            'user_encoder': dataset['user_encoder'],
            'isbn_encoder': dataset['isbn_encoder'],
            'location_encoder': dataset.get('location_encoder'),
            'top_locations': dataset.get('top_locations')
        }
        
        return train_data, test_data