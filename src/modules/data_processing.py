"""
Robust Data Processing Module.
Focuses on preventing NaN values.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self):
        """Initialize the DataProcessor."""
        pass
        
    def load_data(self, data_path):
        """
        Load data from a single CSV file.
        
        Args:
            data_path (str): Path to the preprocessed data CSV
            
        Returns:
            DataFrame: The loaded data
        """
        try:
            print(f"Loading data from {data_path}")
            df = pd.read_csv(data_path)
            print(f"Loaded data shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
    
    def validate_data(self, df):
        """
        Validate the loaded data.
        
        Args:
            df (DataFrame): The dataset to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        print("Validating data...")
        # Check for required columns
        required_cols = ['User-ID', 'Book-Rating', 'Book-Title', 'Book-Author', 
                         'Year-Of-Publication', 'Publisher', 'Age']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            print(f"Available columns: {df.columns.tolist()}")
            return False
                
        return True
    
    def preprocess_data(self, df):
        """
        Feature engineering function that creates normalized and derived features for model training.
        
        Args:
            df (DataFrame): The dataset to preprocess
                
        Returns:
            DataFrame: Processed dataset ready for model training
        """
        # Make a copy to avoid modifying the original
        data = df.copy()
        
        # Fill missing values with appropriate defaults
        data['State'] = data['State'].fillna('Unknown')
        data['Country'] = data['Country'].fillna('Unknown')
        
        # Handle age with fallback value
        median_age = data['Age'].median()
        if pd.isna(median_age):
            median_age = 30
        data['Age'] = data['Age'].fillna(median_age)
        
        # Handle Year-Of-Publication
        try:
            data['Year-Of-Publication'] = pd.to_numeric(data['Year-Of-Publication'], errors='coerce')
            pub_year_median = data['Year-Of-Publication'].median()
            if pd.isna(pub_year_median):
                pub_year_median = 2000
            data['Year-Of-Publication'] = data['Year-Of-Publication'].fillna(pub_year_median)
        except Exception:
            data['Year-Of-Publication'] = 2000
        
        # Create encodings
        data['User-ID-Encoded'] = data['User-ID']
        data['Book-Title-Encoded'] = pd.factorize(data['Book-Title'])[0]
        
        # Create frequency features
        book_counts = data['Book-Title'].value_counts()
        data['Book-Popularity'] = data['Book-Title'].map(book_counts).fillna(1)
        
        author_counts = data['Book-Author'].value_counts()
        data['Author-Popularity'] = data['Book-Author'].map(author_counts).fillna(1)
        
        publisher_counts = data['Publisher'].value_counts()
        data['Publisher-Popularity'] = data['Publisher'].map(publisher_counts).fillna(1)
        
        # Create normalized frequency features
        data['Author-Frequency'] = data['Author-Popularity'] / data['Author-Popularity'].max()
        data['Publisher-Frequency'] = data['Publisher-Popularity'] / data['Publisher-Popularity'].max()
        
        # Create decade feature
        try:
            data['Decade'] = (data['Year-Of-Publication'] // 10 * 10).fillna(2000)
        except Exception:
            data['Decade'] = 2000
        
        # Create normalized features
        # Age normalization
        age_min = data['Age'].min()
        age_max = data['Age'].max()
        if age_min == age_max:
            data['Age-Normalized'] = 0.5
        else:
            data['Age-Normalized'] = (data['Age'] - age_min) / (age_max - age_min)
        
        # Decade normalization
        decade_min = data['Decade'].min()
        decade_max = data['Decade'].max() 
        if decade_min == decade_max:
            data['Decade-Normalized'] = 0.5
        else:
            data['Decade-Normalized'] = (data['Decade'] - decade_min) / (decade_max - decade_min)
        
        # Year normalization
        year_min = data['Year-Of-Publication'].min()
        year_max = data['Year-Of-Publication'].max()
        if year_min == year_max:
            data['Year-Normalized'] = 0.5
        else:
            data['Year-Normalized'] = (data['Year-Of-Publication'] - year_min) / (year_max - year_min)
        
        # Create age groups
        bins = [0, 18, 25, 35, 50, 100, 300]
        labels = [0, 1, 2, 3, 4, 5]
        try:
            data['Age-Group'] = pd.cut(data['Age'], bins=bins, labels=labels).astype(float)
            data['Age-Group'] = data['Age-Group'].fillna(2)
        except Exception:
            data['Age-Group'] = 2
        
        # Create frequency encodings for state and country
        state_counts = data['State'].value_counts(dropna=False)
        data['State-Frequency'] = data['State'].map(state_counts) / len(data)
        data['State-Frequency'] = data['State-Frequency'].fillna(0)
        
        country_counts = data['Country'].value_counts(dropna=False)
        data['Country-Frequency'] = data['Country'].map(country_counts) / len(data)
        data['Country-Frequency'] = data['Country-Frequency'].fillna(0)
        
        # Create rating feature
        data['Normalized-Rating'] = data['Book-Rating'] / 10.0
        
        # Final NaN check and fix
        for column in data.columns:
            if data[column].isna().any():
                if np.issubdtype(data[column].dtype, np.number):
                    data[column] = data[column].fillna(0)
                else:
                    data[column] = data[column].fillna('unknown')
        
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
        print(f"Splitting data with ratio {train_ratio}")
        train_data, test_data = train_test_split(
            data, train_size=train_ratio, random_state=42
        )
        print(f"Train data: {train_data.shape}, Test data: {test_data.shape}")
        return train_data, test_data
    
    def create_training_data(self, data):
        """
        Create training data from preprocessed DataFrame.
        
        Args:
            data (DataFrame): Preprocessed DataFrame with all features
                
        Returns:
            dict: Training dataset
        """
        required_columns = [
            'Age-Normalized', 'Age-Group', 'State-Frequency', 'Country-Frequency',
            'Year-Normalized', 'Author-Frequency', 'Publisher-Frequency', 'Decade',
            'User-ID-Encoded', 'Book-Title-Encoded', 'Normalized-Rating'
        ]
        
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Extract user features
        user_features = np.zeros((len(data), 4))
        user_features[:, 0] = np.nan_to_num(data['Age-Normalized'].values, nan=0.5)
        user_features[:, 1] = np.nan_to_num(data['Age-Group'].values, nan=2.0)
        user_features[:, 2] = np.nan_to_num(data['State-Frequency'].values, nan=0.0)
        user_features[:, 3] = np.nan_to_num(data['Country-Frequency'].values, nan=0.0)
        
        # Extract item features
        item_features = np.zeros((len(data), 4))
        item_features[:, 0] = np.nan_to_num(data['Year-Normalized'].values, nan=0.5)
        item_features[:, 1] = np.nan_to_num(data['Author-Frequency'].values, nan=0.0)
        item_features[:, 2] = np.nan_to_num(data['Publisher-Frequency'].values, nan=0.0)
        item_features[:, 3] = np.nan_to_num(data['Decade'].values / 2020, nan=0.5)
        
        # Get IDs and ratings
        user_ids = data['User-ID-Encoded'].values
        item_ids = data['Book-Title-Encoded'].values
        ratings = np.nan_to_num(data['Normalized-Rating'].values, nan=0.0)
        
        # Create dataset dictionary
        dataset = {
            'user_ids': user_ids,
            'item_ids': item_ids,
            'ratings': ratings,
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