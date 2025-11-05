from sklearn.cluster import KMeans
import numpy as np
from lesson1_pandas import data
from lesson4_principal_component_analysis import get_reduced_words_matrix
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, adjusted_rand_score, silhouette_score
from collections import Counter
from scipy.optimize import linear_sum_assignment

def main():
    MAXIMUM_ITERATIONS = 10
    NUMBER_INITIAL = 2

    data_array = np.array(data)
    number_of_clusters = np.unique(data_array[:, 0]).shape[0]
    print(f"Number of clusters: {number_of_clusters}")
    print("-" * 100)

    reduced_words_matrix = get_reduced_words_matrix()

    kmeans_model = KMeans(
        n_clusters=number_of_clusters,
        max_iter=MAXIMUM_ITERATIONS,
        n_init=NUMBER_INITIAL,
        random_state=0,
        verbose=True
    )
    kmeans_model.fit(reduced_words_matrix)
    print("-" * 100)

    dataframe_centers = pd.DataFrame(
        kmeans_model.cluster_centers_,
        columns=["x", "y"]
    )
    print("Dataframe centers:")
    print(dataframe_centers)
    print("-" * 100)

    predicted_labels = kmeans_model.labels_
    actual_labels = data_array[:, 0].tolist()
    
    # Сопоставление меток кластеров с реальными метками через оптимальное назначение
    unique_clusters = np.sort(np.unique(predicted_labels))
    unique_actual_labels = np.sort(np.unique(actual_labels))

    # Матрица пересечений: строки — кластеры, столбцы — реальные метки
    cluster_index = {cluster: idx for idx, cluster in enumerate(unique_clusters)}
    label_index = {label: idx for idx, label in enumerate(unique_actual_labels)}
    contingency_matrix = np.zeros((len(unique_clusters), len(unique_actual_labels)), dtype=int)

    for actual, predicted in zip(actual_labels, predicted_labels):
        contingency_matrix[cluster_index[predicted], label_index[actual]] += 1

    # Находим оптимальное назначение (максимизируем количество совпадений)
    cost_matrix = -contingency_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    label_mapping = {}
    for row, col in zip(row_ind, col_ind):
        cluster = unique_clusters[row]
        label = unique_actual_labels[col]
        if contingency_matrix[row, col] > 0:
            label_mapping[cluster] = label

    # Для кластеров без соответствий используем наиболее частую метку внутри кластера
    for cluster in unique_clusters:
        if cluster not in label_mapping:
            cluster_indices = np.where(predicted_labels == cluster)[0]
            cluster_actual_labels = [actual_labels[i] for i in cluster_indices]
            if cluster_actual_labels:
                label_mapping[cluster] = Counter(cluster_actual_labels).most_common(1)[0][0]
    
    # Выводим информацию о сопоставлении
    print("Сопоставление кластеров с метками:")
    for cluster, label in sorted(label_mapping.items()):
        print(f"  Кластер {cluster} -> {label}")
    print("-" * 100)
    
    # Преобразуем числовые метки кластеров в строковые для сравнения
    mapped_labels = [label_mapping[label] for label in predicted_labels]
    
    df_labels = pd.DataFrame({
        "Actual": actual_labels,
        "Predicted_Cluster": predicted_labels,
        "Mapped_Label": mapped_labels
    })
    print("Labels comparison (Actual vs Predicted):")
    print(df_labels)
    print("-" * 100)
    
    # Вычисляем метрики качества
    # Adjusted Rand Index (не требует сопоставления меток)
    ari = adjusted_rand_score(actual_labels, predicted_labels)
    print(f"Adjusted Rand Index: {ari:.4f}")
    print("(1.0 = идеальное совпадение, 0.0 = случайное совпадение)")
    
    # Silhouette Score (качество кластеризации по структуре данных)
    silhouette = silhouette_score(reduced_words_matrix, predicted_labels)
    print(f"Silhouette Score: {silhouette:.4f}")
    print("(от -1 до 1, чем выше, тем лучше разделение кластеров)")
    
    # Accuracy после сопоставления меток
    accuracy = accuracy_score(actual_labels, mapped_labels)
    print(f"Accuracy (after label mapping): {accuracy:.4f}")
    print("-" * 100)
    
    print("\nПочему качество может быть низким:")
    print("1. K-Means - это алгоритм БЕЗ учителя: он не знает реальные метки")
    print("2. Потеря информации при PCA: данные сводятся к 2D пространству")
    print("3. Малое количество данных: всего 15 примеров для 3 классов")
    print("4. Возможная плохая разделимость данных в 2D пространстве")
    print("-" * 100)


    plt.figure(figsize=(5,5))

    plt.scatter(reduced_words_matrix[:, 0], reduced_words_matrix[:, 1],
                    c=kmeans_model.labels_, s=50)

    plt.scatter(dataframe_centers["x"], dataframe_centers["y"])

    dy = 0.04

    for index, text in enumerate(kmeans_model.labels_):
        plt.annotate(text, (reduced_words_matrix[index, 0], reduced_words_matrix[index, 1] + dy))
    plt.title("K-Means Clustering")
    plt.show()

if __name__ == "__main__":
    main()