import pandas as pd

df = pd.read_csv(r"C:\Users\marce\Desktop\machine_learning\data\processed_data.csv")

def count_unique_ingredients(df):
    """
    Count the number of unique ingredients in the dataset.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
    
    Returns:
        int: Number of unique ingredients.
    """
    unique_ingredients = set()
    for ingredients in df["ingredients"]:
        ingredient_list = ingredients.split(", ")
        unique_ingredients.update(ingredient_list)
    
    return len(unique_ingredients)

def most_common_ingredients(df):
    """
    Count the occurrences of each ingredient in the dataset.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
    
    Returns:
        dict: Dictionary with ingredient names as keys and their counts as values.
    """
    ingredient_counts = {}
    for ingredients in df["ingredients"]:
        ingredient_list = ingredients.split(", ")
        for ingredient in ingredient_list:
            ingredient = ingredient.strip()
            ingredient_counts[ingredient] = ingredient_counts.get(ingredient, 0) + 1
    
    return ingredient_counts

def count_unique_glass_types(df):
    """
    Count the number of unique glass types in the dataset.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
    
    Returns:
        int: Number of unique glass types.
    """
    return df["glass"].nunique()

def most_common_glass_types(df):
    """
    Get the most common glass types in the dataset.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
    
    Returns:
        pd.Series: Series with glass types and their counts.
    """
    return df["glass"].value_counts()

def count_unique_categories(df):
    """
    Count the number of unique categories in the dataset.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
    
    Returns:
        int: Number of unique categories.
    """
    return df["category"].nunique()

def most_common_categories(df, top_n=10):
    """
    Get the most common categories in the dataset.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        top_n (int): Number of top categories to return.
    
    Returns:
        pd.Series: Series with category names and their counts.
    """
    return df["category"].value_counts().head(top_n)

def count_alcoholic(df):
    """
    Count the number of alcoholic drinks in the dataset.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
    
    Returns:
        int: Count of alcoholic drinks.
    """
    return df["alcoholic"].sum()
