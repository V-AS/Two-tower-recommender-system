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
        Extremely robust preprocessing that guarantees no NaN values.
        
        Args:
            df (DataFrame): The dataset to preprocess
            
        Returns:
            Processed dataset ready for model training
        """
        print("Starting preprocessing...")
        # Make a copy to avoid modifying the original
        data = df.copy()
        
        # Print initial data info and check for NaNs
        print(f"Initial data shape: {data.shape}")
        print(f"NaN values before preprocessing: {data.isna().sum().sum()}")
        
        # STEP 1: Fill missing values with appropriate defaults
        print("Filling missing values...")
        data['State'] = data['State'].fillna('Unknown')
        data['Country'] = data['Country'].fillna('Unknown')
        
        # Calculate median age safely
        median_age = data['Age'].median()
        if pd.isna(median_age):
            median_age = 30  # Fallback if median is NaN
        data['Age'] = data['Age'].fillna(median_age)
        
        # STEP 2: Handle Year-Of-Publication - convert to numeric with error handling
        print("Processing Year-Of-Publication...")
        try:
            data['Year-Of-Publication'] = pd.to_numeric(data['Year-Of-Publication'], errors='coerce')
            # Use safe median or default
            pub_year_median = data['Year-Of-Publication'].median()
            if pd.isna(pub_year_median):
                pub_year_median = 2000  # Safe default
            data['Year-Of-Publication'] = data['Year-Of-Publication'].fillna(pub_year_median)
        except Exception as e:
            print(f"Error processing Year-Of-Publication: {e}")
            # Create a safe fallback
            data['Year-Of-Publication'] = 2000
        
        # STEP 3: Create book encoding safely
        print("Creating encodings...")
        data['User-ID-Encoded'] = data['User-ID']
        data['Book-Title-Encoded'] = pd.factorize(data['Book-Title'])[0]
        
        # STEP 4: Create minimal, safe features that won't generate NaNs
        print("Creating features...")
        
        # Book popularity (safer than complex frequency measures)
        book_counts = data['Book-Title'].value_counts()
        data['Book-Popularity'] = data['Book-Title'].map(book_counts).fillna(1)
        
        # Author popularity (safer implementation)
        author_counts = data['Book-Author'].value_counts()
        data['Author-Popularity'] = data['Book-Author'].map(author_counts).fillna(1)
        
        # Publisher popularity (safer implementation)
        publisher_counts = data['Publisher'].value_counts()
        data['Publisher-Popularity'] = data['Publisher'].map(publisher_counts).fillna(1)
        
        # Normalize these counters to [0,1] range
        data['Author-Frequency'] = data['Author-Popularity'] / data['Author-Popularity'].max()
        data['Publisher-Frequency'] = data['Publisher-Popularity'] / data['Publisher-Popularity'].max()
        
        # STEP 5: Create simple decade feature
        try:
            data['Decade'] = (data['Year-Of-Publication'] // 10 * 10).fillna(2000)
        except Exception as e:
            print(f"Error creating decade feature: {e}")
            data['Decade'] = 2000  # Safe default
        
        # STEP 6: Create basic normalized features with robustness
        print("Creating normalized features...")
        
        # Age normalization with safety
        age_min = data['Age'].min()
        age_max = data['Age'].max()
        if age_min == age_max:  # Avoid division by zero
            data['Age-Normalized'] = 0.5  # Middle value
        else:
            data['Age-Normalized'] = (data['Age'] - age_min) / (age_max - age_min)
        
        # Decade normalization
        decade_min = data['Decade'].min()
        decade_max = data['Decade'].max() 
        if decade_min == decade_max:  # Avoid division by zero
            data['Decade-Normalized'] = 0.5
        else:
            data['Decade-Normalized'] = (data['Decade'] - decade_min) / (decade_max - decade_min)
        
        # Year normalization
        year_min = data['Year-Of-Publication'].min()
        year_max = data['Year-Of-Publication'].max()
        if year_min == year_max:  # Avoid division by zero
            data['Year-Normalized'] = 0.5
        else:
            data['Year-Normalized'] = (data['Year-Of-Publication'] - year_min) / (year_max - year_min)
        
        # STEP 7: Create simple age grouping
        bins = [0, 18, 25, 35, 50, 100]
        labels = [0, 1, 2, 3, 4]
        try:
            data['Age-Group'] = pd.cut(data['Age'], bins=bins, labels=labels).astype(float)
            data['Age-Group'] = data['Age-Group'].fillna(2)  # Middle group as default
        except Exception as e:
            print(f"Error creating age groups: {e}")
            data['Age-Group'] = 2  # Middle group as default
        
        # STEP 8: Create frequency encodings for state and country
        state_counts = data['State'].value_counts(dropna=False)
        data['State-Frequency'] = data['State'].map(state_counts) / len(data)
        data['State-Frequency'] = data['State-Frequency'].fillna(0)
        
        country_counts = data['Country'].value_counts(dropna=False)
        data['Country-Frequency'] = data['Country'].map(country_counts) / len(data)
        data['Country-Frequency'] = data['Country-Frequency'].fillna(0)
        
        # STEP 9: Create rating feature
        data['Normalized-Rating'] = data['Book-Rating'] / 10.0
        
        # Final NaN check and fix
        print(f"NaN values after preprocessing: {data.isna().sum().sum()}")
        if data.isna().sum().sum() > 0:
            print("WARNING: Still have NaN values. Fixing...")
            for column in data.columns:
                if data[column].isna().any():
                    print(f"Column {column} has {data[column].isna().sum()} NaN values")
                    # For numeric columns, use 0
                    if np.issubdtype(data[column].dtype, np.number):
                        data[column] = data[column].fillna(0)
                    # For string columns, use 'unknown'
                    else:
                        data[column] = data[column].fillna('unknown')
        
        print(f"Final data shape: {data.shape}")
        print(f"Final NaN check: {data.isna().sum().sum()}")
        
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
        Create bare-minimum training data from preprocessed DataFrame.
        
        Args:
            data (DataFrame): Preprocessed DataFrame with all features
                
        Returns:
            dict: Training dataset
        """
        print("Creating training data...")
        # Make sure these columns exist
        required_columns = [
            'Age-Normalized', 'Age-Group', 'State-Frequency', 'Country-Frequency',
            'Year-Normalized', 'Author-Frequency', 'Publisher-Frequency', 'Decade',
            'User-ID-Encoded', 'Book-Title-Encoded', 'Normalized-Rating'
        ]
        
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            print(f"ERROR: Missing columns: {missing}")
            print(f"Available columns: {data.columns.tolist()}")
            raise ValueError(f"Missing required columns: {missing}")
        
        # Extract user features - just 4 basic features
        user_features = np.zeros((len(data), 4))  # Pre-allocate array
        
        # Fill each column individually with error checking
        try:
            user_features[:, 0] = data['Age-Normalized'].values
        except Exception as e:
            print(f"Error filling Age-Normalized: {e}")
            user_features[:, 0] = 0.5  # Default value
            
        try:
            user_features[:, 1] = data['Age-Group'].values
        except Exception as e:
            print(f"Error filling Age-Group: {e}")
            user_features[:, 1] = 2  # Default middle group
            
        try:
            user_features[:, 2] = data['State-Frequency'].values
        except Exception as e:
            print(f"Error filling State-Frequency: {e}")
            user_features[:, 2] = 0  # Default value
            
        try:
            user_features[:, 3] = data['Country-Frequency'].values
        except Exception as e:
            print(f"Error filling Country-Frequency: {e}")
            user_features[:, 3] = 0  # Default value
        
        # Extract item features - just 4 basic features
        item_features = np.zeros((len(data), 4))  # Pre-allocate array
        
        # Fill each column individually with error checking
        try:
            item_features[:, 0] = data['Year-Normalized'].values
        except Exception as e:
            print(f"Error filling Year-Normalized: {e}")
            item_features[:, 0] = 0.5  # Default value
            
        try:
            item_features[:, 1] = data['Author-Frequency'].values
        except Exception as e:
            print(f"Error filling Author-Frequency: {e}")
            item_features[:, 1] = 0  # Default value
            
        try:
            item_features[:, 2] = data['Publisher-Frequency'].values
        except Exception as e:
            print(f"Error filling Publisher-Frequency: {e}")
            item_features[:, 2] = 0  # Default value
            
        try:
            item_features[:, 3] = data['Decade'].values / 2020  # Normalize by max possible year
        except Exception as e:
            print(f"Error filling Decade: {e}")
            item_features[:, 3] = 0.5  # Default value
            
        # Check for NaN values and fix
        if np.isnan(user_features).any():
            print(f"WARNING: NaN in user_features: {np.isnan(user_features).sum()}")
            user_features = np.nan_to_num(user_features)
            
        if np.isnan(item_features).any():
            print(f"WARNING: NaN in item_features: {np.isnan(item_features).sum()}")
            item_features = np.nan_to_num(item_features)
        
        # Create dataset with error checking
        try:
            user_ids = data['User-ID-Encoded'].values
        except Exception as e:
            print(f"Error getting User-ID-Encoded: {e}")
            user_ids = np.arange(len(data))
            
        try:
            item_ids = data['Book-Title-Encoded'].values
        except Exception as e:
            print(f"Error getting Book-Title-Encoded: {e}")
            item_ids = np.arange(len(data))
            
        try:
            ratings = data['Normalized-Rating'].values
        except Exception as e:
            print(f"Error getting Normalized-Rating: {e}")
            ratings = data['Book-Rating'].values / 10.0  # Alternative calculation
        
        # Final check for NaN values in ratings
        if np.isnan(ratings).any():
            print(f"WARNING: NaN in ratings: {np.isnan(ratings).sum()}")
            ratings = np.nan_to_num(ratings)
        
        dataset = {
            'user_ids': user_ids,
            'item_ids': item_ids,
            'ratings': ratings,
            'user_features': user_features,
            'item_features': item_features
        }
        
        print("Training data created successfully.")
        print(f"User features shape: {user_features.shape}")
        print(f"Item features shape: {item_features.shape}")
        print(f"Ratings shape: {ratings.shape}")
        
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