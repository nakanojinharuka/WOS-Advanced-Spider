import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from transformers import pipeline, BertTokenizer
from bertopic import BERTopic
import nltk
from sklearn.decomposition import LatentDirichletAllocation, NMF
nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words("english"))


# 1 加载数据
def load_data(file_paths):
    """加载CSV文件并合并"""
    return pd.concat([pd.read_csv(file) for file in file_paths], ignore_index=True)


# 2 文本预处理
def preprocess_texts(df):
    text_cols = ['abstract', 'title'] + [f'keyword{i}' for i in range(1, 25)]
    texts = []
    for col in text_cols:
        if col in df.columns:
            texts.extend(df[col].dropna().astype(str).tolist())
    return texts


# 3 主题提取
def bert_topic_modeling(texts, num_topics=8):
    topic_model = BERTopic()
    topics, _ = topic_model.fit_transform(texts)
    return topic_model.get_topic_info().head(num_topics)


def lda_topic_modeling(texts, num_topics=8):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for idx, topic in enumerate(lda.components_):
        topics.append(f"主题 {idx+1}: " + ", ".join([feature_names[i] for i in topic.argsort()[-10:]]))
    return topics


def nmf_topic_modeling(texts, num_topics=8):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    nmf = NMF(n_components=num_topics, random_state=42)
    nmf.fit(X)
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for idx, topic in enumerate(nmf.components_):
        topics.append(f"主题 {idx+1}: " + ", ".join([feature_names[i] for i in topic.argsort()[-10:]]))
    return topics


# 4 主流程
def main(file_paths):
    df = load_data(file_paths)
    texts = preprocess_texts(df)
    topics = lda_topic_modeling(texts)
    print("主题提取结果：", topics)


if __name__ == '__main__':
    main(['C:/Workspace/WOS-Advanced-Spider/Filtered_Sorted/filtered_2024_urban_mobility_patterns1.csv',
          'C:/Workspace/WOS-Advanced-Spider/Filtered_Sorted/filtered_2024_urban_mobility_public_transport1.csv'])
