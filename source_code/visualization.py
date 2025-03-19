import quantitative_analysis as quan
import qualitative_analysis as qual
import matplotlib.pyplot as plt

def unique_numbers(df):
    print(f"\nLiczba unikalnych składników: {quan.count_unique_ingredients(df)}")
    print(f"\nLiczba unikalnych typów szkła: {quan.count_unique_glass_types(df)}")
    print(f"\nLiczba unikalnych kategorii: {quan.count_unique_categories(df)}")
    print(f"\nLiczba koktajli alkoholowych: {quan.count_alcoholic(df)}")

def charts_ingredients(df):
    all_counts = quan.most_common_ingredients(df)
    sorted_counts = sorted(all_counts.items(), key=lambda x: x[1], reverse=True)
    top10 = sorted_counts[:10]
    others = sorted_counts[10:]
    others_total = sum(count for _, count in others)
    labels = [ingredient for ingredient, count in top10]
    sizes = [count for ingredient, count in top10]
    labels.append("Inne")
    sizes.append(others_total)
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("Procentowy udział składników (Top 10 i reszta)")
    plt.axis('equal')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, sizes, color='skyblue')
    plt.xlabel("Składniki")
    plt.ylabel("Liczba wystąpień")
    plt.title("Top 10 składników oraz reszta")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def charts_glasses(df):
    counts_all = quan.most_common_glass_types(df)
    labels = counts_all.index.tolist()
    sizes = counts_all.values.tolist()
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("Procentowy udział typów szkła (wszystkie)")
    plt.axis('equal')
    plt.show()
    
    plt.figure(figsize=(12, 6))
    counts_all.plot(kind='bar', color='skyblue')
    plt.xlabel("Typ szkła")
    plt.ylabel("Liczba wystąpień")
    plt.title("Wszystkie typy szkła")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def logistic_regression():
    print("Regresja logistyczna - dokładność:", qual.accuracy_log)
    print("Raport klasyfikacji (Regresja logistyczna):")
    print(qual.classification_report(qual.y_test, qual.y_pred_log, zero_division=0))
    cm_log = qual.confusion_matrix(qual.y_test, qual.y_pred_log)
    print("Macierz pomyłek (Regresja logistyczna):")
    print(cm_log)

def random_forest__regression():
    print("Random forest - accuracy:", qual.accuracy_rf)
    print("Raport klasyfikacji (Random forest):")
    print(qual.classification_report(qual.y_test, qual.y_pred_rf, zero_division=0))
    cm_rf = qual.confusion_matrix(qual.y_test, qual.y_pred_rf)
    print("Macierz pomyłek (Random forest):")
    print(cm_rf)