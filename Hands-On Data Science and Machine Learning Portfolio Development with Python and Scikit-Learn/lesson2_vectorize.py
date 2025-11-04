from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# Явно указываем, что можно импортировать
__all__ = [
    'process_text',
    'get_weights',
    'get_similarity',
    'get_all_results'
]


def _get_default_sentences():
    """Возвращает примеры предложений для обработки."""
    return [
        "Machine Learning (ML) is a core subfield of Artificial Intelligence (AI) that enables computers to learn patterns from data and improve their performance on a task over time—without being explicitly programmed. It is what powers recommendations on Netflix, fraud detection in banking, and language understanding in chatbots.",
        "At its core, machine learning is about building systems that can: Ingest data, Identify patterns or structure, Make decisions or predictions, Improve with more data and experience",
        "Arthur Samuel defined machine learning as the field of study that gives computers the ability to learn without being explicitly programmed.",
        "Machine learning methods are typically divided into three major categories: supervised learning, unsupervised learning, and reinforcement learning.",
        "Supervised learning is the machine learning task of learning a function that maps an input to a output based on example input-output pairs.",
        "Unsupervised learning is the machine learning task of learning a function to map inputs to no labels based on example inputs without labeled outputs.",
        "Reinforcement learning is the machine learning task of learning what action to take under what circumstances.",
        "The process of learning from data is called training. A model that has been trained on a set of data is called a trained model.",
        "A model that has not been trained on any data is called a model.",
        "A model that has been trained on a set of data is called a trained model."
    ]


def process_text(sentences=None, verbose=False):
    """
    Обрабатывает текстовые данные с использованием CountVectorizer и TF-IDF.
    
    Args:
        sentences (list, optional): Список предложений для обработки.
            Если None, используются примеры по умолчанию.
        verbose (bool, optional): Если True, выводит промежуточные результаты.
            По умолчанию False.
    
    Returns:
        dict: Словарь с результатами обработки:
            - 'bag_of_words': DataFrame с bag-of-words представлением
            - 'tfidf_matrix': DataFrame с TF-IDF весами
            - 'weights': DataFrame с весами слов и их частотами
            - 'similarity': DataFrame с матрицей косинусного сходства слов
            - 'feature_names': numpy array с названиями признаков
    """
    if sentences is None:
        sentences = _get_default_sentences()
    
    if verbose:
        print("Sentences:")
        print(sentences)
        print("-" * 100)
    
    # Векторизация
    vectorizer = CountVectorizer()
    vectorizer.fit(sentences)
    feature_names = vectorizer.get_feature_names_out()
    
    if verbose:
        print("Feature names:")
        print(feature_names)
        print("-" * 100)
    
    # Bag of words
    bag_of_words = vectorizer.transform(sentences)
    dataframe_bag_of_words = pd.DataFrame(bag_of_words.todense(), columns=feature_names)
    
    if verbose:
        print("Dataframe bag of words:")
        print(dataframe_bag_of_words)
        print("-" * 100)
    
    # TF-IDF трансформация
    transformer = TfidfTransformer()
    words_matrix = transformer.fit_transform(bag_of_words)
    dataframe_words_matrix = pd.DataFrame(words_matrix.todense(), columns=feature_names)
    
    if verbose:
        print("Dataframe words matrix:")
        print(dataframe_words_matrix)
        print("-" * 100)
    
    # Подсчет весов и частот
    word_counts = np.asarray(bag_of_words.sum(axis=0)).ravel().tolist()
    weights = np.asarray(words_matrix.mean(axis=0)).ravel().tolist()
    
    dataframe_counts = pd.DataFrame({"word": feature_names, "count": word_counts})
    dataframe_weights = pd.DataFrame({"word": feature_names, "weight": weights})
    
    dataframe_weights = dataframe_weights.merge(dataframe_counts, on="word", how="left")
    
    if verbose:
        print("Dataframe weights:")
        print(dataframe_weights)
        print("-" * 100)
    
    # Матрица сходства слов
    word_similarity = cosine_similarity(words_matrix.T, words_matrix.T)
    dataframe_word_similarity = pd.DataFrame(
        word_similarity, 
        columns=feature_names, 
        index=feature_names
    )
    
    if verbose:
        print("Dataframe word similarity:")
        print(dataframe_word_similarity)
        print("-" * 100)
    
    return {
        "bag_of_words": dataframe_bag_of_words,
        "tfidf_matrix": dataframe_words_matrix,
        "weights": dataframe_weights,
        "similarity": dataframe_word_similarity,
        "feature_names": feature_names
    }


def get_weights(sentences=None):
    """
    Возвращает DataFrame с весами слов и их частотами.
    
    Args:
        sentences (list, optional): Список предложений для обработки.
            Если None, используются примеры по умолчанию.
    
    Returns:
        pd.DataFrame: DataFrame с колонками 'word', 'weight', 'count'
    """
    results = process_text(sentences, verbose=False)
    return results["weights"]


def get_similarity(sentences=None):
    """
    Возвращает матрицу косинусного сходства между словами.
    
    Args:
        sentences (list, optional): Список предложений для обработки.
            Если None, используются примеры по умолчанию.
    
    Returns:
        pd.DataFrame: DataFrame с матрицей сходства (слова x слова)
    """
    results = process_text(sentences, verbose=False)
    return results["similarity"]


def get_all_results(sentences=None, verbose=False):
    """
    Псевдоним для process_text. Возвращает все результаты обработки.
    
    Args:
        sentences (list, optional): Список предложений для обработки.
        verbose (bool, optional): Если True, выводит промежуточные результаты.
    
    Returns:
        dict: Словарь со всеми результатами обработки
    """
    return process_text(sentences, verbose)


def main():
    """Основная функция для запуска скрипта напрямую."""
    print("=" * 100)
    print("Обработка текстовых данных с векторизацией и TF-IDF")
    print("=" * 100)
    print()
    
    # Запускаем обработку с выводом всех промежуточных результатов
    results = process_text(verbose=True)
    
    print("\n" + "=" * 100)
    print("Обработка завершена!")
    print("=" * 100)


if __name__ == "__main__":
    main()
