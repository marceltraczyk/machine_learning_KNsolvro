import pandas as pd

def load_data(json_path):
    """
    Load JSON data into a pandas DataFrame.
    
    Parameters:
        json_path (str): Path to the JSON file.
    
    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    return pd.read_json(json_path)

def remove_unnecessary_columns(df):
    """
    Remove unnecessary columns from the DataFrame.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame without unnecessary columns.
    """
    columns_to_drop = ["imageUrl", "createdAt", "updatedAt", "instructions"]
    return df.drop(columns=columns_to_drop, errors="ignore")

def extract_ingredient_names(df):
    """
    Extract only the names of ingredients from the 'ingredients' column.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame with extracted ingredient names.
    """
    df["ingredients"] = df["ingredients"].apply(
        lambda ingredients: ", ".join([ingredient["name"] for ingredient in ingredients])
    )
    return df

def handle_nan_values(df):
    """
    Replace NaN values in the 'tags' column with an empty list.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame with NaN values handled.
    """
    df["tags"] = df["tags"].apply(lambda x: ", ".join(x if isinstance(x, list) else []))
    return df

def save_data(df, output_path):
    """
    Save DataFrame to a CSV file.
    
    Parameters:
        df (pd.DataFrame): DataFrame to save.
        output_path (str): Path to save the CSV file.
    """
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    # Load and preprocess data
    df = load_data("data/cocktail_dataset.json")
    df = remove_unnecessary_columns(df)
    df = handle_nan_values(df)
    df = extract_ingredient_names(df)
    
    # Save processed data to CSV
    save_data(df, "data/processed_data.csv")
