import pandas as pd
from transformers import BertTokenizer
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib import rcParams

# 1. 定义屏蔽词列表（可以自定义或扩展）
stopwords = ['the', 'and', 'of', 'in', 'to', 'for', 'a', 'on', 'is', 'with', 'at', 'this', 'an', 's', 'es', 'neo',
             'urban', 'renewal', 'simulation', 'city', 'project', 'base', 'model', 'study', 'analysis', 'cities',
             'development', 'redevelopment', 'process', 'public', 'result', 'provide', 'area', 'research', 'policies',
             'method', 'building', 'change', 'planning', 'datum', 'data', 'design', 'high', 'low', 'paper', 'making'
             'effect', 'case', 'system', 'propose', 'impact', 'strategy', 'new', 'factor', 'level', 'different',
             'value', 'approach', 'use', 'utilize', 'increase', 'decrease', 'build', 'space', 'finding', 'old',
             'young', 'plan', 'significant', 'important', 'framework', 'explore', 'relationship', 'key', 'property',
             'keywords', 'keyword', 'author', 'plus', 'multi', 'learn', 'information', 'population', 'algorithm',
             'diffusion', 'game', 'social', 'community', 'china', 'policy', 'politic', 'right', 'spatial', 'race',
             'politics', 'participate', 'participation', 'rights', 'state']
# 2. 加载 BERT 分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
stopwords_subwords = set()
for word in stopwords:
    # 将每个屏蔽词转换为子词
    subwords = tokenizer.tokenize(word)
    stopwords_subwords.update(subwords)  # 更新子词集合
# 3. 定义分词函数
def bert_tokenize(text):
    """
    使用 BERT 分词器进行分词，并过滤停用词。
    """
    # 使用 BERT 进行子词级分词
    tokens = tokenizer.tokenize(text)
    # 保留字母组成的词，排除屏蔽词
    filtered_tokens = [token for token in tokens if token.isalpha() and token not in stopwords_subwords]
    return filtered_tokens
# 4. 定义main函数
def main_proceed():
    # 5. 加载数据
    df = pd.read_csv("sorted/sorted_urban_renewal.csv")  # 替换为实际文件路径
    # 6. 指定需要处理的文本列（支持多列）
    columns_to_process = [f'keyword{i}' for i in range(22)]  # 替换为你的列名列表
    # 7. 处理多列文本数据
    all_texts = []  # 存储所有列的分词结果
    for col in columns_to_process:
        if col in df.columns:
            # 获取所有文本并去除空值
            texts = df[col].dropna()
            # 应用 BERT 分词
            processed_texts = texts.apply(bert_tokenize).apply(lambda x: ' '.join(x))  # 转成字符串形式
            all_texts.extend(processed_texts)
    # 8. 提取二元短语 (bigrams)
    vectorizer = CountVectorizer(ngram_range=(1, 2))  # 设置 ngram_range 为 (2, 2) 表示只提取二元短语
    X = vectorizer.fit_transform(all_texts)
    # 9. 获取二元短语及其词频
    word_freq = dict(zip(vectorizer.get_feature_names_out(), X.toarray().sum(axis=0)))
    # 10. 排序词频
    sorted_word_freq = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True))
    return sorted_word_freq

if __name__=='__main__':
    # 11. 运行main函数
    word_freq_list = main_proceed()
    # 12. 输出前30个最常见的词
    print("前30个最常见的词频：", dict(list(word_freq_list.items())[:30]))
    # 13. 绘制词频图
    rcParams['font.sans-serif'] = ['SimHei', 'Times New Roman']  # 可以根据需要换成其他字体
    rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.figure(figsize=(16, 10))
    plt.bar(list(word_freq_list.keys())[:30], list(word_freq_list.values())[:30])  # 显示前30个
    plt.xlabel("词语")
    plt.ylabel("频率")
    plt.title("词频统计（BERT 分词）")
    plt.xticks(rotation=30)
    plt.show()
