"""
Simplified Data Processing Module for loading and preprocessing data.
Handles dataset validation and splitting.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class DataProcessor:
    def __init__(self):
        """Initialize the DataProcessor."""
        self.user_encoder = LabelEncoder()
        self.book_title_encoder = LabelEncoder()
        self.author_encoder = LabelEncoder()
        self.publisher_encoder = LabelEncoder()
        self.state_encoder = LabelEncoder()
        self.country_encoder = LabelEncoder()
        
    def load_data(self, data_path):
        """
        Load data from a single CSV file.
        
        Args:
            data_path (str): Path to the preprocessed data CSV
            
        Returns:
            DataFrame: The loaded data
            
        Raises:
            IOError: If file cannot be read
        """
        try:
            df = pd.read_csv(data_path)
            return df
        except pd.errors.EmptyDataError:
            raise ValueError("The CSV file is empty")
        except pd.errors.ParserError:
            raise ValueError("Error parsing CSV format")
        except Exception as e:
            raise IOError(f"Failed to load data: {str(e)}")
    
    def validate_data(self, df):
        """
        Validate the loaded data.
        
        Args:
            df (DataFrame): The dataset to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check for required columns
            required_cols = ['User-ID', 'Book-Rating', 'Book-Title', 'Book-Author', 
                             'Year-Of-Publication', 'Publisher', 'Age']
            
            if not all(col in df.columns for col in required_cols):
                return False
                
            return True
        except Exception:
            return False
    
    def preprocess_data(self, df):
        """
        Preprocess the loaded data.
        
        Args:
            df (DataFrame): The dataset to preprocess
            
        Returns:
            Processed dataset ready for model training
        """
        # Make a copy to avoid modifying the original
        data = df.copy()
        
        # Fill missing values with appropriate defaults
        data['State'] = data['State'].fillna('Unknown')
        data['Country'] = data['Country'].fillna('Unknown')
        
        # Convert Year-Of-Publication to numeric if possible
        data['Year-Of-Publication'] = pd.to_numeric(data['Year-Of-Publication'], errors='coerce')
        data['Year-Of-Publication'] = data['Year-Of-Publication'].fillna(data['Year-Of-Publication'].median())
        
        # Normalize Age
        data['Age-Normalized'] = (data['Age'] - data['Age'].min()) / (data['Age'].max() - data['Age'].min())
        
        # Normalize Book-Rating to [0, 1] range
        data['Normalized-Rating'] = data['Book-Rating'] / 10.0
        
        # Encode categorical variables
        data['User-ID-Encoded'] = self.user_encoder.fit_transform(data['User-ID'])
        data['Book-Title-Encoded'] = self.book_title_encoder.fit_transform(data['Book-Title'])
        data['Author-Encoded'] = self.author_encoder.fit_transform(data['Book-Author'])
        data['Publisher-Encoded'] = self.publisher_encoder.fit_transform(data['Publisher'])
        data['State-Encoded'] = self.state_encoder.fit_transform(data['State'])
        data['Country-Encoded'] = self.country_encoder.fit_transform(data['Country'])
        
        # Create author and publisher popularity features
        author_counts = data['Book-Author'].value_counts()
        publisher_counts = data['Publisher'].value_counts()
        
        data['Author-Popularity'] = data['Book-Author'].map(author_counts) / author_counts.max()
        data['Publisher-Popularity'] = data['Publisher'].map(publisher_counts) / publisher_counts.max()
        
        # Normalize Year-Of-Publication to [0, 1] range
        data['Year-Normalized'] = (data['Year-Of-Publication'] - data['Year-Of-Publication'].min()) / \
                                  (data['Year-Of-Publication'].max() - data['Year-Of-Publication'].min() + 1e-10)
        
        return data
    
    def split_data(self, data, train_ratio=0.8):
        """
        Split the dataset into training and testing sets.
        
        Args:
            data (DataFrame): The dataset to split
            train_ratio (float): Ratio of training data
            
        Returns:
            train_data, test_data: The split datasets
            
        Raises:
            ValueError: If train_ratio is not in (0, 1)
        """
        if train_ratio <= 0 or train_ratio >= 1:
            raise ValueError("train_ratio must be in (0, 1)")
        
        # Split the data
        train_data, test_data = train_test_split(
            data, train_size=train_ratio, random_state=42
        )
        
        return train_data, test_data
    
    def create_training_data(self, data):
        """
        Create training data from preprocessed DataFrame.
        
        Args:
            data (DataFrame): Preprocessed DataFrame with all encoded features
                
        Returns:
            dict: Training dataset
        """
        # Extract user features
        user_features = data[[
            'Age-Normalized', 
            'State-Encoded', 
            'Country-Encoded'
        ]].values
        
        # Extract item features
        item_features = data[[
            'Year-Normalized',
            'Author-Popularity',
            'Publisher-Popularity',
            'Author-Encoded',
            'Publisher-Encoded'
        ]].values
        
        # Create the training dataset dictionary
        dataset = {
            'user_ids': data['User-ID-Encoded'].values,
            'item_ids': data['Book-Title-Encoded'].values,
            'ratings': data['Normalized-Rating'].values,
            'user_features': user_features,
            'item_features': item_features
        }
        
        return dataset
    
    def get_book_mapping(self, data):
        """
        Create a mapping from encoded book IDs to book details.
        
        Args:
            data (DataFrame): Preprocessed data
            
        Returns:
            dict: Mapping from encoded book IDs to book details
        """
        book_mapping = {}
        
        for _, row in data.drop_duplicates('Book-Title-Encoded').iterrows():
            book_mapping[row['Book-Title-Encoded']] = {
                'title': row['Book-Title'],
                'author': row['Book-Author'],
                'year': row['Year-Of-Publication'],
                'publisher': row['Publisher']
            }
        
        return book_mapping