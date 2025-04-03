# tests/unit/test_data_processing.py
"""
Unit test for Data Processing Module (M2).
"""
import sys
import os
import unittest
import pandas as pd
import numpy as np
import tempfile

# Add project root to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.modules.data_processing import DataProcessor


class TestDataProcessor(unittest.TestCase):
    """Test cases for the DataProcessor module."""

    def setUp(self):
        """Set up test environment before each test method."""
        self.data_processor = DataProcessor()

        # Create a sample DataFrame for testing
        self.sample_data = pd.DataFrame(
            {
                "User-ID": [1, 2, 3, 4, 5],
                "Book-Rating": [5, 8, 3, 9, 6],
                "Book-Title": ["Title1", "Title2", "Title3", "Title4", "Title5"],
                "Book-Author": ["Author1", "Author2", "Author3", "Author4", "Author5"],
                "Year-Of-Publication": [2000, 2005, 2010, 2015, 2020],
                "Publisher": ["Pub1", "Pub2", "Pub3", "Pub4", "Pub5"],
                "Age": [25, 30, 35, 40, 45],
                "State": ["State1", "State2", "State3", "State4", "State5"],
                "Country": ["Country1", "Country2", "Country3", "Country4", "Country5"],
            }
        )

        # Create a temporary file for data loading tests
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        self.sample_data.to_csv(self.temp_file.name, index=False)

    def tearDown(self):
        """Clean up after each test method."""
        # Delete temporary file
        if hasattr(self, "temp_file") and os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_load_data(self):
        """Test loading data from a CSV file."""
        # Load data from temporary file
        loaded_data = self.data_processor.load_data(self.temp_file.name)

        # Check that loaded data has the right shape
        self.assertEqual(loaded_data.shape, self.sample_data.shape)

        # Check that loaded data has the right columns
        for col in self.sample_data.columns:
            self.assertIn(col, loaded_data.columns)

    def test_validate_data_valid(self):
        """Test validating a valid dataset."""
        # Validate sample data
        is_valid = self.data_processor.validate_data(self.sample_data)

        # Check validation result
        self.assertTrue(is_valid)

    def test_validate_data_invalid(self):
        """Test validating an invalid dataset with missing columns."""
        # Create invalid data by dropping a required column
        invalid_data = self.sample_data.drop(columns=["Book-Rating"])

        # Validate invalid data
        is_valid = self.data_processor.validate_data(invalid_data)

        # Check validation result
        self.assertFalse(is_valid)

    def test_preprocess_data(self):
        """Test data preprocessing functionality."""
        # Preprocess sample data
        processed_data = self.data_processor.preprocess_data(self.sample_data)

        # Check that expected derived columns are created
        expected_new_columns = [
            "Book-Title-Encoded",
            "Book-Popularity",
            "Author-Popularity",
            "Publisher-Popularity",
            "Author-Frequency",
            "Publisher-Frequency",
            "Decade",
            "Age-Normalized",
            "Year-Normalized",
            "State-Frequency",
            "Country-Frequency",
            "Normalized-Rating",
        ]

        for col in expected_new_columns:
            self.assertIn(col, processed_data.columns)

        # Check that Age-Normalized is between 0 and 1
        self.assertTrue((processed_data["Age-Normalized"] >= 0).all())
        self.assertTrue((processed_data["Age-Normalized"] <= 1).all())

        # Check that no NaN values remain
        self.assertFalse(processed_data.isnull().any().any())

    def test_preprocess_data_with_missing_values(self):
        """Test preprocessing with missing values."""
        # Create data with missing values
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[0, "Age"] = np.nan
        data_with_missing.loc[1, "State"] = np.nan
        data_with_missing.loc[2, "Country"] = np.nan

        # Preprocess data with missing values
        processed_data = self.data_processor.preprocess_data(data_with_missing)

        # Check that no NaN values remain
        self.assertFalse(processed_data.isnull().any().any())

    def test_split_data(self):
        """Test splitting data into training and testing sets."""
        # Split data
        train_data, test_data = self.data_processor.split_data(
            self.sample_data, train_ratio=0.6
        )

        # Check the split proportions (with some allowance for rounding)
        expected_train_size = int(len(self.sample_data) * 0.6)
        self.assertAlmostEqual(len(train_data), expected_train_size, delta=1)
        self.assertAlmostEqual(
            len(test_data), len(self.sample_data) - expected_train_size, delta=1
        )

        # Check that the combined data has the same size as the original
        self.assertEqual(len(train_data) + len(test_data), len(self.sample_data))

    def test_create_training_data(self):
        """Test creating training data."""
        # First preprocess the data
        processed_data = self.data_processor.preprocess_data(self.sample_data)

        # Create training data
        training_data = self.data_processor.create_training_data(processed_data)

        # Check that the training data has the expected structure
        expected_keys = [
            "user_ids",
            "item_ids",
            "ratings",
            "user_features",
            "item_features",
        ]
        for key in expected_keys:
            self.assertIn(key, training_data)

        # Check that arrays have the expected shapes
        self.assertEqual(len(training_data["user_ids"]), len(processed_data))
        self.assertEqual(len(training_data["item_ids"]), len(processed_data))
        self.assertEqual(len(training_data["ratings"]), len(processed_data))
        self.assertEqual(training_data["user_features"].shape, (len(processed_data), 4))
        self.assertEqual(training_data["item_features"].shape, (len(processed_data), 4))

    def test_create_training_data_missing_columns(self):
        """Test creating training data with missing required columns."""
        # Create data missing a required column
        missing_col_data = self.sample_data.copy()
        missing_col_data = self.data_processor.preprocess_data(missing_col_data)
        # Remove a required column that's created during preprocessing
        missing_col_data = missing_col_data.drop(columns=["Age-Normalized"])

        # Try to create training data from invalid data
        with self.assertRaises(ValueError):
            self.data_processor.create_training_data(missing_col_data)

    def test_get_book_mapping(self):
        """Test creating book mapping."""
        # Preprocess data first
        processed_data = self.data_processor.preprocess_data(self.sample_data)

        # Get book mapping
        book_mapping = self.data_processor.get_book_mapping(processed_data)

        # Check that mapping contains all unique books
        unique_books = processed_data["Book-Title-Encoded"].nunique()
        self.assertEqual(len(book_mapping), unique_books)

        # Check that mapping has the right structure
        for book_id in book_mapping:
            book_details = book_mapping[book_id]
            self.assertIn("title", book_details)
            self.assertIn("author", book_details)
            self.assertIn("year", book_details)
            self.assertIn("publisher", book_details)


if __name__ == "__main__":
    unittest.main()
