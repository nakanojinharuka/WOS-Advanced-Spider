import pandas as pd
import spacy
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib import rcParams
import re


def extract_noun_phrases(text):
    """
    使用spaCy提取文本中的名词短语。
    """
    doc = nlp(text)
    # 提取名词短语
    return [chunk.text.lower() for chunk in doc.noun_chunks]


def combined_tokenize(text):
    # 这个正则表达式将会提取所有的字母和数字
    regex_tokens = re.findall(r'\b\w+\b', text.lower())
    # 使用 spaCy 对正则分词结果进行进一步清理
    doc = nlp(' '.join(regex_tokens))  # 将分词结果拼接为字符串并传入 spaCy 处理
    # 获取 spaCy 分词结果，去除停用词和标点符号
    spacy_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    # 进一步去除屏蔽词
    filtered_tokens = [token for token in spacy_tokens if token not in stopwords]
    return filtered_tokens

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
                 'young', 'plan', 'significant', 'important', 'framework', 'explore', 'relationship', 'key', 'property',
                 'keywords', 'keyword', 'author', 'plus', 'multi', 'learn', 'information', 'population', 'algorithm',
                 'diffusion', 'game', 'social', 'community', 'china', 'policy', 'politic', 'right', 'spatial']  # 示例屏蔽词
    # 3. 加载数据
    df = pd.read_csv("sorted/sorted_urban_renewal.csv")  # 根据实际情况替换路径
    # 4. 指定文本数据列
    columns_to_process = [f'keyword{i}' for i in range(21)]
    # 5. 处理多列文本数据
    all_tokens = []
    noun_phrases = []
    for col in columns_to_process:
        # 确保当前列存在且不是空值
        if col in df.columns:
            texts = df[col].dropna()  # 移除缺失值
            # 提取名词短语
            noun_phrases.extend(texts.apply(extract_noun_phrases).sum())  # 汇总名词短语
            # 分词处理
            tokens = texts.apply(combined_tokenize).sum()  # 汇总分词结果
            all_tokens.extend(tokens)
    # 6. 词频统计 (使用 CountVectorizer)
    vectorizer = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)
    X = vectorizer.fit_transform([all_tokens])
    word_freq = dict(zip(vectorizer.get_feature_names_out(), X.toarray()[0]))

    # 7. 排序词频并显示前40个
    sorted_word_freq = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True))
    top_40_words = dict(list(sorted_word_freq.items())[:40])

    # 8. 输出词频
    print("词频前40名：", top_40_words)

    # 9. 绘制词频图
    # 设置中文字体，确保中文显示正常
    rcParams['font.sans-serif'] = ['SimHei','Times New Roman']  # 可以根据需要换成其他字体
    rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.figure(figsize=(25, 16))
    plt.bar(top_40_words.keys(), top_40_words.values())
    plt.xlabel("词语")
    plt.ylabel("频率")
    plt.title("前40个词频")
    plt.xticks(rotation=45)
    plt.show()
