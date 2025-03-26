import pandas as pd


def clean_data(file_path):
    df = pd.read_csv(file_path)
    print(f"Original data shape: {df.shape}")
    df.drop_duplicates()

    # Drop rows with missing values
    df.dropna(inplace=True)
    print(f"Cleaned data shape: {df.shape}")

    return df


def store_cleaned_data(df, file_path):
    df.to_csv(file_path, index=False)

# matching IDs between ratings and users/books
# remove rows with the IDs that are not in the other dataset
def match_ids(ratings, users, books):
    # ratings user id = users user id
    ratings = ratings[ratings["User-ID"].isin(users["User-ID"])]
    # ratings ISBN = books ISBN
    ratings = ratings[ratings["ISBN"].isin(books["ISBN"])]
    return ratings


if __name__ == "__main__":
    ratings_path = "data/raw/Ratings.csv"
    ratings_df = clean_data(ratings_path)
    books_path = "data/raw/Books.csv"
    books_df = clean_data(books_path)
    users_path = "data/raw/Users.csv"
    users_df = clean_data(users_path)
    
    ratings_df = match_ids(ratings_df, users_df, books_df)
    store_cleaned_data(ratings_df, "data/processed/Ratings.csv")
    store_cleaned_data(books_df, "data/processed/Books.csv")
    store_cleaned_data(users_df, "data/processed/Users.csv")
