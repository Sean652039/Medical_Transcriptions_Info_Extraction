import pandas as pd
import re
import os
import matplotlib.pyplot as plt

def analyze_context(text):
    # Using regular expression to find words surrounding numbers
    matches = re.findall(r'(\b(?:year|yr)s?[- ]old\b)', text)

    # Counting the occurrences of words surrounding numbers
    context_counts = {}
    for match in matches:
        if match in context_counts:
            context_counts[match] += 1
        else:
            context_counts[match] = 1

    return context_counts

if __name__ == "__main__":
    # read data
    samples_add = '../mtsamples.csv'
    context = pd.read_csv(samples_add)['transcription']

    # Analyzing words surrounding numbers in each line of text
    results = []
    for text in context:
        counts = analyze_context(str(text))
        results.append(counts)

    # Converting results to a DataFrame
    results_df = pd.DataFrame(results)

    top_words = results_df.sum().sort_values(ascending=True)

    # Plotting the distribution
    plt.figure(figsize=(10, 6))
    top_words.plot(kind='bar')
    plt.xlabel('Context')
    plt.ylabel('Count')
    plt.title('Most Frequent Words')

    # Creating the 'plot' directory if it does not exist
    if not os.path.exists('../plot'):
        os.makedirs('../plot')

    # Saving the plot to the 'plot' directory
    plt.savefig('../plot/words_distribution.png')

    # Displaying the plot
    plt.show()

    # Printing the top 10 frequent words
    print(top_words)