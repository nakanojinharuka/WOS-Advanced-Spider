import pandas as pd
import spacy
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib import rcParams
import re
if __name__=='__main__':
    # 1. 加载 spaCy 英语模型
    nlp = spacy.load("en_core_web_sm")
    # 2. 定义屏蔽词列表（需要屏蔽的词汇）
    stopwords = ['the', 'and', 'of', 'in', 'to', 'for', 'a', 'on', 'is', 'with', 'at', 'this', 'an', 's',
                 'urban', 'renewal', 'simulation', 'city', 'project', 'base', 'model', 'study', 'analysis',
                 'development', 'redevelopment', 'process', 'public', 'result', 'provide', 'area', 'research',
                 'method', 'building', 'change', 'planning', 'datum', 'data', 'design', 'high', 'low', 'paper',
                 'effect', 'case', 'system', 'propose', 'impact', 'strategy', 'new', 'factor', 'level', 'different',
                 'value', 'approach', 'use', 'utilize', 'increase', 'decrease', 'build', 'space', 'finding', 'old',
                 'young', 'plan', 'significant', 'important', 'framework', 'explore', 'relationship', 'keywordsurban',
                 'keywords', 'keyword', 'author', 'plus', 'multi', 'learn', 'information', 'population', 'algorithm',
                 'diffusion', 'game']  # 示例屏蔽词
    # 你可以根据需要添加更多词汇到这个列表中
    # 加载数据
    df = pd.read_csv("urban_renewal_simulation.csv")  # 根据实际情况替换路径

    # 3. 提取文本数据
    texts = df['keywords'].dropna()  # 获取"Text"列，并删除空值

    def extract_noun_phrases(text):
        """
        使用spaCy提取文本中的名词短语。
        """
        doc = nlp(text)
        # 提取名词短语
        return [chunk.text.lower() for chunk in doc.noun_chunks]

    def combined_tokenize(text):
        # 使用正则表达式进行初步分词
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # 小写和大写字母之间插入空格
        text = re.sub(r'([A-Z])([A-Z][a-z])', r'\1 \2', text)  # 处理连续的大写字母（如 HTML)
        text = re.sub(r'\d+', ' ', text)  # 去掉数字（如果需要）
        # 这个正则表达式将会提取所有的字母和数字
        regex_tokens = re.findall(r'\b\w+\b', text.lower())

        # 使用 spaCy 对正则分词结果进行进一步清理
        doc = nlp(' '.join(regex_tokens))  # 将分词结果拼接为字符串并传入 spaCy 处理

        # 获取 spaCy 分词结果，去除停用词和标点符号
        spacy_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

        # 进一步去除屏蔽词
        filtered_tokens = [token for token in spacy_tokens if token not in stopwords]

        return filtered_tokens

    # 5. 对文本进行分词处理
    noun_phrases = texts.apply(extract_noun_phrases)
    processed_texts = texts.apply(combined_tokenize)
    # 6. 词频提取 (使用 CountVectorizer)
    vectorizer = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)
    X = vectorizer.fit_transform(processed_texts)
    word_freq = dict(zip(vectorizer.get_feature_names_out(), X.sum(axis=0).tolist()[0]))

    # 7. 排序词频并显示前10个
    sorted_word_freq = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True))
    top_10_words = dict(list(sorted_word_freq.items())[:30])

    # 8. 输出词频
    print("词频前30名：", top_10_words)

    # 9. 绘制词频图
    # 设置中文字体，确保中文显示正常
    rcParams['font.sans-serif'] = ['SimHei','Times New Roman']  # 可以根据需要换成其他字体
    rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.figure(figsize=(10, 6))
    plt.bar(top_10_words.keys(), top_10_words.values())
    plt.xlabel("词语")
    plt.ylabel("频率")
    plt.title("前30个词频")
    plt.xticks(rotation=-45)
    plt.show()
