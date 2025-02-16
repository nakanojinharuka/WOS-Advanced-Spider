import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from bertopic import BERTopic
from top2vec import Top2Vec
from transformers import pipeline
import gensim
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import argparse
import nltk

nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words("english"))

class TopicExtractor:
    def __init__(self, num_topics=5):
        self.num_topics = num_topics

    def preprocess_texts(self, df):
        """提取文本列并合并"""
        text_cols = ['abstract', 'title'] + [f'keyword{i}' for i in range(1, 27)]
        texts = []
        for col in text_cols:
            if col in df.columns:
                texts.extend(df[col].dropna().astype(str).tolist())
        return texts

    def lda_topic_modeling(self, texts):
        """使用 LDA 进行主题建模"""
        vectorizer = CountVectorizer(stop_words='english')
        X = vectorizer.fit_transform(texts)
        lda = LatentDirichletAllocation(n_components=self.num_topics, random_state=42)
        lda.fit(X)

        feature_names = vectorizer.get_feature_names_out()
        topics = [
            f"主题 {idx+1}: " + ", ".join([feature_names[i] for i in topic.argsort()[-10:]])
            for idx, topic in enumerate(lda.components_)
        ]
        return topics

    def nmf_topic_modeling(self, texts):
        """使用 NMF 进行主题建模"""
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(texts)
        nmf = NMF(n_components=self.num_topics, random_state=42)
        nmf.fit(X)

        feature_names = vectorizer.get_feature_names_out()
        topics = [
            f"主题 {idx+1}: " + ", ".join([feature_names[i] for i in topic.argsort()[-10:]])
            for idx, topic in enumerate(nmf.components_)
        ]
        return topics

    def bert_topic_modeling(self, texts):
        """使用 BERTopic 进行主题建模"""
        topic_model = BERTopic(nr_topics=self.num_topics)
        topics, _ = topic_model.fit_transform(texts)
        return topic_model.get_topic_info().head(self.num_topics)

    def top2vec_topic_modeling(self, texts):
        """使用 Top2Vec 进行主题建模"""
        model = Top2Vec(texts, speed="deep-learn", workers=16)
        topic_words, _, _ = model.get_topics(self.num_topics)
        return ["主题 {}: {}".format(i+1, ", ".join(topic)) for i, topic in enumerate(topic_words)]

    def gpt_topic_extraction(self, texts, model="gpt-3.5-turbo"):
        """使用 GPT 从文本中提取主题"""
        topic_generator = pipeline("text-generation", model=model)
        combined_text = " ".join(texts)[:1000]  # 限制文本大小
        result = topic_generator(f"请从以下文本中提取5个核心主题: {combined_text}", max_length=100, do_sample=False)
        return result[0]['generated_text']

    def zero_shot_topic_extraction(self, texts, candidate_labels=None):
        """使用 Zero-Shot 方法进行主题提取"""
        if candidate_labels is None:
            candidate_labels = ["社会", "科技", "环境", "经济", "健康"]
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        results = [classifier(text, candidate_labels) for text in texts[:10]]  # 处理前10条
        return results

    def extract_topics(self, texts, method="all"):
        """统一主题提取方法"""
        topic_results = {}

        if method == "all" or method == "lda":
            topic_results["LDA"] = self.lda_topic_modeling(texts)
        if method == "all" or method == "nmf":
            topic_results["NMF"] = self.nmf_topic_modeling(texts)
        if method == "all" or method == "bertopic":
            topic_results["BERTopic"] = self.bert_topic_modeling(texts)
        if method == "all" or method == "top2vec":
            topic_results["Top2Vec"] = self.top2vec_topic_modeling(texts)
        if method == "all" or method == "gpt":
            topic_results["GPT"] = self.gpt_topic_extraction(texts)
        if method == "all" or method == "zero-shot":
            topic_results["Zero-Shot"] = self.zero_shot_topic_extraction(texts)

        return topic_results

def main(file_paths, method="all", num_topics=5):
    df = pd.concat([pd.read_csv(file) for file in file_paths], ignore_index=True)
    extractor = TopicExtractor(num_topics)
    texts = extractor.preprocess_texts(df)

    topic_results = extractor.extract_topics(texts, method)

    print("主题提取结果：")
    for key, value in topic_results.items():
        print(f"\n🔹 {key} 方法：")
        for topic in value:
            print(topic)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--files', nargs='+', default=["C:/Workspace/WOS-Advanced-Spider/Filtered_Sorted/filtered_2024_mobility_metropolitan_area1.csv"], help='CSV文件路径')
    parser.add_argument('-m', '--method', choices=['all', 'lda', 'nmf', 'bertopic', 'top2vec', 'gpt', 'zero-shot'], default="lda", help='选择主题提取方法')
    parser.add_argument('-n', '--num_topics', type=int, default=9, help='主题数量')
    args = parser.parse_args()
    main(args.files, args.method, args.num_topics)
