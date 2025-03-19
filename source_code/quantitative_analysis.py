import pandas as pd

df = pd.read_csv(r"C:\Users\marce\Desktop\machine_learning\data\processed_data.csv")

# checking the unique number of components 
def count_unique_ingredients(df):
    unique_ingredients = set()
    for ingredients in df["ingredients"]:
        ingredient = ingredients.split(", ")
        unique_ingredients.update(ingredient)
    
    return len(unique_ingredients)

# counts of the occurrence of each component
def most_common_ingredients(df):
    ingredient_counts = {}
    for ingredients in df["ingredients"]:
        ingredients = ingredients.split(", ")
        for ingredient in ingredients:
            ingredient_counts[ingredient.strip()] = ingredient_counts.get(ingredient, 0) + 1
    return ingredient_counts

def count_unique_glass_types(df):
    return df["glass"].nunique()

def most_common_glass_types(df):
    return df["glass"].value_counts()

def count_unique_categories(df):
    return df["category"].nunique()

def most_common_categories(df, top_n=10):
    return df["category"].value_counts().head(top_n)

def count_alcoholic(df):
    return df["alcoholic"].sum()