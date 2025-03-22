import quantitative_analysis as quan
import qualitative_analysis as qual
import matplotlib.pyplot as plt

def unique_numbers(df):
    print(f"\nNumber of unique components: {quan.count_unique_ingredients(df)}")
    print(f"\nNumber of unique glass types: {quan.count_unique_glass_types(df)}")
    print(f"\nNumber of unique categories: {quan.count_unique_categories(df)}")
    print(f"\nNumber of alcoholic cocktails: {quan.count_alcoholic(df)}")

def charts_ingredients(df):
    all_counts = quan.most_common_ingredients(df)
    sorted_counts = sorted(all_counts.items(), key=lambda x: x[1], reverse=True)
    top10 = sorted_counts[:10]
    others = sorted_counts[10:]
    others_total = sum(count for _, count in others)
    labels = [ingredient for ingredient, count in top10]
    sizes = [count for ingredient, count in top10]
    labels.append("Others")
    sizes.append(others_total)
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("Percentage of components (Top 10 and the others)")
    plt.axis('equal')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, sizes, color='skyblue')
    plt.xlabel("Ingredients")
    plt.ylabel("Number of appearances")
    plt.title("Top 10 ingredients and the others")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def charts_glasses(df):
    counts_all = quan.most_common_glass_types(df)
    labels = counts_all.index.tolist()
    sizes = counts_all.values.tolist()
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("Percentage of glass types (all)")
    plt.axis('equal')
    plt.show()
    
    plt.figure(figsize=(12, 6))
    counts_all.plot(kind='bar', color='skyblue')
    plt.xlabel("Glass type")
    plt.ylabel("Number of appearances")
    plt.title("All types of glass")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def logistic_regression():
    print("Logistic regression - accuracy:", qual.accuracy_log)
    print("Classification Report (Logistic Regression):")
    print(qual.classification_report(qual.y_test, qual.y_pred_log, zero_division=0))
    cm_log = qual.confusion_matrix(qual.y_test, qual.y_pred_log)
    print("Confusion Matrix (Logistic Regression):")
    print(cm_log)

def random_forest__regression():
    print("Random forest - accuracy:", qual.accuracy_rf)
    print("Classification report (Random forest):")
    print(qual.classification_report(qual.y_test, qual.y_pred_rf, zero_division=0))
    cm_rf = qual.confusion_matrix(qual.y_test, qual.y_pred_rf)
    print("Confusion matrix (Random forest):")
    print(cm_rf)