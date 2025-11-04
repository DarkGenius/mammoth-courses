from lesson2_vectorize import get_weights
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def main():
    df_weights = get_weights()
    word_frequency = pd.Series(df_weights["count"])
    word_frequency.index = df_weights["word"]
    dictionary_word_frequency = word_frequency.to_dict()

    print("-" * 100)
    print("Dictionary word frequency:")
    for key, value in dictionary_word_frequency.items():
        print(f"{key}: {value}")
    print("-" * 100)

    word_cloud = WordCloud(width=2000, height=2000).generate_from_frequencies(dictionary_word_frequency)
    
    plt.imshow(word_cloud)
    plt.axis("off")
    plt.show()
 
if __name__ == "__main__":
    main()
