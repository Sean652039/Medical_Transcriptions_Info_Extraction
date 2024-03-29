# Medical Transcriptions Extraction (Age and Treatments)

This project aims to extract patient age and treatment plans from the [Medical Transcriptions dataset](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions?resource=download) (Medical transcription data scraped from mtsamples.com). Two strategies and models are employed for extracting the two target pieces of information.

- **Age Extraction**: Age is extracted using regular expressions as pseudo-labels for supervised learning. The model chosen for this task is a pre-trained BERT model.
- **Treatments Extraction**: Unsupervised clustering is used for treatment plan extraction, with the model being LDA.

## Directory Structure

```bash
├───mtssamples
├───utils
│   ├───analysis
│   └───stopwords
├───scripts
```

The main code file for the project is `main.py`. Files under `utils` mainly focus on text analysis, word frequency analysis for constructing stopwords.

## Usage

- This project is based on the Python programming language, utilizing external libraries such as nltk, PyTorch, etc. It supports Apple Silicon acceleration. The program is run using Python version 3.10. Setting up Python environment using [Anaconda](https://www.anaconda.com/) is recommended.

```bash
python main.py -n num_topics -f file_path
```

`num_topics` is the number of topics for the LDA model used in treatment plan extraction, defaulting to 5.

`file_path` is the path to the text information needing extraction.

## Age Extraction

Firstly, the text is analyzed using regular expressions in `utils/ContextAnalysis.py` to determine the format of ages mentioned in the records. The results are as follows.

![words_distribution](https://cdn.jsdelivr.net/gh/Sean652039/pic_bed@main/uPic/words_distribution.png)

Based on these results, pseudo-labels are generated for age extraction.

The choice of using the BERT model is justified for the following reasons:
1. **Bidirectional Contextual Understanding**: BERT, based on the Transformer architecture, considers both left and right context simultaneously. This bidirectionality enables BERT to comprehensively understand the context of words, capturing semantic and syntactic structures better.
2. **Pre-training and Fine-tuning**: BERT is first pre-trained on large-scale text data in an unsupervised manner, learning universal language representations. These pre-trained representations can be fine-tuned on specific tasks, making them adaptable to various natural language processing tasks such as text classification, named entity recognition, question answering, etc.
3. **Excellent Performance**: BERT has demonstrated outstanding performance on various NLP tasks, surpassing previous feature-engineering methods and other pre-trained models in many tasks.
4. **Interpretability and Scalability**: Due to its Transformer-based architecture, BERT offers a degree of interpretability, allowing understanding of the model's decision-making process through attention weights, etc. Furthermore, BERT's model structure is relatively flexible, allowing for adaptation to specific tasks or datasets by adding more layers or modifying parameters.

## Treatment Plan Extraction

Initially, all texts undergo word frequency analysis in `utils/StopwordsCreation.py`, setting words appearing more than 60 times as stopwords. The final result is merged with nltk's stopwords and stored in `stopwords/custom_stopwords.txt`.

LDA model is chosen for treatment plan extraction. LDA, Latent Dirichlet Allocation, is a probabilistic generative model for topic modeling of text data. LDA's basic assumption is that each document can be viewed as a mixture of different topics, and each topic can be viewed as a probability distribution of different words. By analyzing the occurrence of words in documents, LDA attempts to infer the hidden topic structure. (However, the results may not meet expectations.)

Five topics are set for each patient's record to be clustered separately using LDA (epochs set to 1000). After obtaining the results, the latter part of the words in each topic is merged and subjected to word frequency analysis, outputting the top 5 words with the highest frequency, which likely contain treatment plans. **This strategy is based on observation, thus the effectiveness may vary.**