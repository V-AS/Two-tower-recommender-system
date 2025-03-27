"""
Enhanced Data Processing Module with better feature engineering.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class DataProcessor:
    def __init__(self):
        """Initialize the DataProcessor."""
        self.categorical_encoders = {}
        self.scalers = {}
        
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
        Enhanced preprocessing with better feature engineering.
        
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
        data['Age'] = data['Age'].fillna(data['Age'].median())
        
        # Handle Year-Of-Publication - convert to numeric and fill missing
        data['Year-Of-Publication'] = pd.to_numeric(data['Year-Of-Publication'], errors='coerce')
        data['Year-Of-Publication'] = data['Year-Of-Publication'].fillna(data['Year-Of-Publication'].median())
        
        # Create categorical features - using frequency encoding for high cardinality categories
        # Author popularity (frequency encoding)
        author_counts = data['Book-Author'].value_counts()
        data['Author-Frequency'] = data['Book-Author'].map(author_counts) / len(data)
        
        # Publisher popularity (frequency encoding)
        publisher_counts = data['Publisher'].value_counts()
        data['Publisher-Frequency'] = data['Publisher'].map(publisher_counts) / len(data)
        
        # Country popularity (frequency encoding)
        country_counts = data['Country'].value_counts()
        data['Country-Frequency'] = data['Country'].map(country_counts) / len(data)
        
        # State/Province popularity (frequency encoding)
        state_counts = data['State'].value_counts()
        data['State-Frequency'] = data['State'].map(state_counts) / len(data)
        
        # Age buckets (create age groups for better generalization)
        bins = [0, 18, 25, 35, 50, 100]
        labels = [0, 1, 2, 3, 4]
        data['Age-Group'] = pd.cut(data['Age'], bins=bins, labels=labels)
        
        # Create decade groups for books
        data['Decade'] = (data['Year-Of-Publication'] // 10) * 10
        
        # Create normalized features
        scaler = StandardScaler()
        
        # Normalize Age
        data['Age-Normalized'] = (data['Age'] - data['Age'].min()) / (data['Age'].max() - data['Age'].min() + 1e-6)
        
        # Normalize Book-Rating to [0, 1] range
        data['Normalized-Rating'] = data['Book-Rating'] / 10.0
        
        # Normalize Year-Of-Publication
        data['Year-Normalized'] = (data['Year-Of-Publication'] - data['Year-Of-Publication'].min()) / \
                               (data['Year-Of-Publication'].max() - data['Year-Of-Publication'].min() + 1e-6)
        
        # One-hot encode low-cardinality categoricals
        # For users - keep the original fields but also add encoded versions
        data['User-ID-Encoded'] = data['User-ID']
        data['Book-Title-Encoded'] = pd.factorize(data['Book-Title'])[0]
        
        # Create book ratios - what percentage of all books by this author has the user read?
        author_book_counts = data.groupby('Book-Author')['Book-Title'].transform('nunique')
        data['Author-Book-Ratio'] = 1 / author_book_counts
        
        # Generate interaction features (combinations of important features)
        data['Age-Country-Interaction'] = data['Age-Normalized'] * data['Country-Frequency']
        data['Year-Publisher-Interaction'] = data['Year-Normalized'] * data['Publisher-Frequency']
        
        return data
    
    def split_data(self, data, train_ratio=0.8):
        """
        Split the dataset into training and testing sets.
        
        Args:
            data (DataFrame): The dataset to split
            train_ratio (float): Ratio of training data
            
        Returns:
            train_data, test_data: The split datasets
        """
        train_data, test_data = train_test_split(
            data, train_size=train_ratio, random_state=42
        )
        
        return train_data, test_data
    
    def create_training_data(self, data):
        """
        Create enhanced training data from preprocessed DataFrame.
        
        Args:
            data (DataFrame): Preprocessed DataFrame with all features
                
        Returns:
            dict: Training dataset
        """
        # Extract user features - focus on the most informative features
        user_features = np.column_stack((
            data['Age-Normalized'].values,
            data['Age-Group'].values,
            data['State-Frequency'].values,
            data['Country-Frequency'].values
        ))
        
        # Extract item features - focus on the most informative features
        item_features = np.column_stack((
            data['Year-Normalized'].values,
            data['Author-Frequency'].values,
            data['Publisher-Frequency'].values,
            data['Decade'].values
        ))
        
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