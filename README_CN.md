# 医疗记录信息提取（年龄和治疗方案）

本项目针对数据集Medical Transcriptions（Medical transcription data scraped from mtsamples.com）对病人的年龄和治疗方案进行提取。对于两个目标信息， 运用了两种策略和模型。

- 年龄，通过正则表达式提取年龄作为伪标签，进行监督学习，模型选用预训练的BERT模型。
- 治疗方案，采用无监督聚类，模型为LDA

## 文件结构

```bash
├───mtssamples
├───utils
│   ├───analysis
│   └───stopwords
├───scripts
```

项目的主要代码文件为 `main.py`. `utils`下包含的文件主要针对对文本的分析和基于文本进行词频统计来构建停用词。

## 运行指南

- 本项目基于 Python 编程语言，用到的外部代码库主要包括nltk, PyTorch等。支持Apple Silicon加速。程序运行使用的 Python 版本为 3.10，建议使用 [Anaconda](https://www.anaconda.com/) 配置 Python 环境。

```bash
python main.py -n num_topics -f file_path
```

num_topics是提取治疗方案是LDA模型的主题数目，默认是5。

file_path是所需要提取的文本信息。

## 年龄提取

首先通过正则表达式对文本进行分析`utils/ContextAnalysis.py`， 统计记录中所用的年龄格式。结果如下所示。

![words_distribution](https://cdn.jsdelivr.net/gh/Sean652039/pic_bed@main/uPic/words_distribution.png)

通过该结果对文本进行标签提取制作伪标签。

模型选择上选择了预训练的BERT模型，原因如下：

1. **双向上下文理解**：BERT是一种基于Transformer的双向语言模型，它能够同时考虑左侧和右侧的上下文信息。这种双向性使得BERT在理解语言时能够更全面地考虑单词的语境，从而更好地捕捉语义和句法结构。
2. **预训练与微调**：BERT首先在大规模的文本数据上进行了无监督的预训练，从而学习到了通用的语言表示。这些预训练的表示可以通过微调在特定任务上进行进一步的训练，从而适应于不同的自然语言处理任务，如文本分类、命名实体识别、问答等。
3. **性能优异**：BERT在各种自然语言处理任务上取得了优异的性能，包括但不限于文本分类、文本生成、句子相似度计算等。在很多任务上，BERT的表现已经超过了之前的基于特征工程的方法以及其他预训练模型。
4. **可解释性与可扩展性**：由于BERT是基于Transformer架构的，它具有一定的可解释性，可以通过注意力权重等方式来理解模型的决策过程。此外，BERT的模型结构相对灵活，可以通过添加更多的层或者修改参数来适应特定的任务或者数据集。

## 治疗方案提取

首先对所有文本进行词频统计`utils/StopwordsCreation.py`，对于出现60次以上的单词设置为停用词。将最终结果和nltk的stopwords合并并存在`stopwords/custom_stopwords.txt`.

模型选择上选择了LDA，LDA是Latent Dirichlet Allocation的缩写，是一种用于文本数据主题建模的概率生成模型。LDA的基本假设是每个文档可以被看作是不同主题的混合，而每个主题又可以被看作是不同单词的概率分布。通过分析文档中的单词出现情况，LDA试图推断出隐藏的主题结构。（但是效果不如预期）

本项目设置了5个主题，针对每一个病人的记录单独进行聚类（对每个病人的情况单独用LDA聚类，epoch为1000）。得到结果后，对每个主题下后半部分的词语进行合并并进行词频统计，输出频率最高的5个单词，其中大概率包含治疗方案。**此策略是基于观察所得，所以效果不是特别好。**