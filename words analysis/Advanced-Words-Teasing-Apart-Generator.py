import pandas as pd
from transformers import BertTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import XLNetTokenizer
from transformers import RobertaTokenizer, RobertaForMaskedLM
import nltk
from nltk.corpus import stopwords
import glob
import os
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import gensim
from gensim.models import Word2Vec
import numpy as np


# 1. 加载停用词列表
def load_stopwords(file_path):
    with open(file_path, 'r') as file:
        return set(line.strip().lower() for line in file.readlines())


# 2. 加载数据（加载文件并返回 DataFrame）
def load_data(file_keyword, path=r'filtered sorted'):
    all_files = [f for f in glob.glob(os.path.join(path, '*.csv')) if file_keyword in os.path.basename(f)]
    dataframes = []
    for file in all_files:
        df = pd.read_csv(file)
        dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)


# 3. 文本预处理：提取文本列并进行分词和停用词过滤
def process_texts(df, stopwords_subwords):
    all_texts = []
    columns_to_process = ['abstract', 'title'] + [f'keyword{i}' for i in range(1, 30)]
    for col in columns_to_process:
        if col in df.columns:
            texts = df[col].dropna()
            processed_texts = texts.apply(lambda x: bert_tokenize(x, stopwords_subwords)).apply(lambda x: ' '.join(x))
            all_texts.extend(processed_texts)
    return all_texts


# 4. 使用 BERT Tokenizer 和停用词过滤
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
# tokenizer_t5 = T5Tokenizer.from_pretrained('t5-large')
# tokenizer_xlnet = XLNetTokenizer.from_pretrained('xlnet-base-cased')
# tokenizer_roberta = RobertaTokenizer.from_pretrained('roberta-large')

def bert_tokenize(text, stopwords_subwords):
    tokens = tokenizer.tokenize(text)
    filtered_tokens = [token for token in tokens if token.isalpha() and token.lower() not in stopwords_subwords]
    return filtered_tokens
# def t5_tokenize(text, stopwords_subwords):
#     tokens = tokenizer_t5.tokenize(text)
#     filtered_tokens = [token for token in tokens if token.isalpha() and token.lower() not in stopwords_subwords]
#     return filtered_tokens
# def xlnet_tokenize(text, stopwords_subwords):
#     tokens = tokenizer_xlnet.tokenize(text)
#     filtered_tokens = [token for token in tokens if token.isalpha() and token.lower() not in stopwords_subwords]
#     return filtered_tokens
# def roberta_tokenize(text, stopwords_subwords):
#     tokens = tokenizer_roberta.tokenize(text)
#     filtered_tokens = [token for token in tokens if token.isalpha() and token.lower() not in stopwords_subwords]
#     return filtered_tokens

# 5. 使用 TF-IDF 计算停用词
def generate_tfidf_stopwords(texts, min_tfidf_threshold=0.1):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = np.array(X.sum(axis=0)).flatten()
    word_scores = dict(zip(feature_names, tfidf_scores))

    # 选择 TF-IDF 值较低的词作为候选停用词
    stopwords_tfidf = {word for word, score in word_scores.items() if score < min_tfidf_threshold}
    return stopwords_tfidf


# 6. 使用 Word2Vec 查找与高频词相似的词
def generate_word2vec_stopwords(texts, model=None, min_similarity=0.7):
    if model is None:
        model = Word2Vec(sentences=[text.split() for text in texts], vector_size=100, window=5, min_count=1, workers=4)

    stopwords_word2vec = set()
    for word in model.wv.index_to_key:
        similar_words = model.wv.most_similar(word, topn=10)
        for similar_word, similarity in similar_words:
            if similarity >= min_similarity:
                stopwords_word2vec.add(similar_word)

    return stopwords_word2vec


# 7. 合并和优化停用词列表
def combine_stopwords(file_stopwords, tfidf_stopwords, w2v_stopwords):
    combined_stopwords = file_stopwords.union(tfidf_stopwords).union(w2v_stopwords)
    return combined_stopwords


# 8. 计算词频分析
def word_freq_analysis(file_keyword, gram_numbers, stopwords_subwords, min_word_freq=10, min_bigram_freq=8):
    # 加载数据
    df = load_data(file_keyword)

    # 处理文本并生成停用词
    all_texts = process_texts(df, stopwords_subwords)

    # 进行词频分析
    vectorizer = CountVectorizer(ngram_range=(gram_numbers, gram_numbers))
    X = vectorizer.fit_transform(all_texts)
    word_freq = dict(zip(vectorizer.get_feature_names_out(), X.toarray().sum(axis=0)))

    # Filter based on frequency
    filtered_word_freq = {phrase: freq for phrase, freq in word_freq.items() if
                          (gram_numbers == 1 and freq >= min_word_freq) or (
                                      gram_numbers == 2 and freq >= min_bigram_freq)}

    # Filter out stopwords2 from bigrams
    sorted_word_freq = dict(sorted(filtered_word_freq.items(), key=lambda item: item[1], reverse=True))

    return sorted_word_freq


# 9. Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-k', '--file_keyword', required=True, help='Keywords for your *.csv files')
parser.add_argument('-g', '--gram_numbers', required=True, type=int, choices=[1, 2, 3],
                    help='Unigram: 1, Bigram: 2, Trigram: 3')
args = parser.parse_args()

# 加载和生成停用词
files = ["file1.csv", "file2.csv"]  # 替换为你自己的文件路径
df = load_data(args.file_keyword)  # 使用-k参数来加载文件

# 生成基于 TF-IDF 和 Word2Vec 的停用词
texts = df['abstract'].dropna().tolist()  # 以 'abstract' 列为例
tfidf_stopwords = generate_tfidf_stopwords(texts)
w2v_stopwords = generate_word2vec_stopwords(texts)

# 加载领域相关停用词文件
file_stopwords = load_stopwords(f'temp/stopwords/stopwords.txt')

# 合并所有停用词
final_stopwords = combine_stopwords(file_stopwords, tfidf_stopwords, w2v_stopwords)

# 将合并后的停用词应用到分词和分析中
stopwords_subwords = final_stopwords

# Main processing
if __name__ == '__main__':
    word_freq_list = word_freq_analysis(args.file_keyword, args.gram_numbers, stopwords_subwords)
    print("Top 30 word frequencies:", dict(list(word_freq_list.items())[:30]))
    pd.DataFrame(list(word_freq_list.items()), columns=['Phrase', 'Frequency']).to_csv(
        f"frequency_tables/filtered_{args.file_keyword}_GA{args.gram_numbers}.csv", index=False)
