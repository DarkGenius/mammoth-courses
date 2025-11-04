import pandas as pd
import numpy as np

def main():
    # Create a list of data
    data = [
        ["Neutral",  "This was fine."],
        ["Neutral",  "This was OK."],
        ["Negative", "This was bad."],
        ["Negative", "I did not like this."],
        ["Positive", "This was good."],
        ["Positive", "I loved this."]
    ]

    # Create a dataframe
    dataframe = pd.DataFrame(data, columns=["Label", "Review"])
    print("Dataframe:")
    print(dataframe)
    print("-" * 100)

    # Get the labels
    labels = dataframe["Label"]
    print("Labels:")
    print(labels)
    print("-" * 100)

    number_of_labels = np.unique(labels).shape[0]
    print("Number of labels:")
    print(number_of_labels)
    print("-" * 100)

    print("Unique labels:")
    print(np.unique(labels))
    print("-" * 100)

if __name__ == "__main__":
    main()
