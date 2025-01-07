import pandas as pd
from transformers import BertTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import XLNetTokenizer
from transformers import RobertaTokenizer, RobertaForMaskedLM
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib import rcParams
import nltk
from nltk.corpus import stopwords
# 1. 定义屏蔽词列表（可以自定义或扩展）
stopwords1 = ['the', 'and', 'of', 'in', 'to', 'for', 'a', 'on', 'is', 'with', 'at', 'this', 'an', 's', 'es', 'neo',
              'index', 'people', 'across', 'groups', 'influence', 'central', 'first', 'second', 'third', 'higher',
              'urban', 'renewal', 'simulation', 'city', 'project', 'base', 'model', 'study', 'analysis', 'cities',
              'services', 'non', 'types', 'levels', 'there', 'object', 'orient', 'oriented', 'horizontal', 'vertical',
              'development', 'redevelopment', 'process', 'public', 'result', 'provide', 'area', 'research', 'policies',
              'into', 'segmentation', 'segment', 'geographical', 'distribute', 'distributed', 'many', 'much', 'item',
              'method', 'building', 'change', 'planning', 'datum', 'data', 'design', 'high', 'low', 'paper', 'making',
              'local', 'small', 'region', 'regions', 'datum', 'locations', 'because', 'cause', 'since', 'when',
              'effect', 'case', 'system', 'propose', 'impact', 'strategy', 'new', 'factor', 'level', 'different',
              'quality', 'network', 'good', 'better', 'best', 'per', 'persist', 'percent', 'items', 'decide',
              'value', 'approach', 'use', 'utilize', 'increase', 'decrease', 'build', 'space', 'finding', 'old',
              'cellular', 'middle', 'bad', 'worse', 'worst', 'complex', 'simple', 'simply', 'very', 'mono',
              'young', 'plan', 'significant', 'important', 'framework', 'explore', 'relationship', 'key', 'property',
              'wr', 'medium', 'deserve', 'innovation', 'create', 'creation', 'provides', 'provided', 'properties',
              'keywords', 'keyword', 'author', 'plus', 'multi', 'learn', 'information', 'population', 'algorithm',
              'auto', 'whose', 'economic', 'gap', 'identify', 'role', 'act', 'effective', 'sheet',
              'diffusion', 'game', 'social', 'community', 'china', 'policy', 'politic', 'right', 'spatial', 'race',
              'impacts', 'efficient', 'media', 'systems', 'play', 'evaluate', 'future', 'everyone',
              'politics', 'participate', 'participation', 'rights', 'state', 'gen', 'post', 'projects', 'land', 'cover',
              'two', 'four', 'could', 'would', 'should', 'reveal', 'evaluation', 'decision', 'accessible',
              'neural', 'patterns', 'pattern', 'eva', 'ag', 'learning', 'risk', 'that', 'by', 'from', 'was', 'as',
              'results', 'analyze', 'frontier', 'less', 'least', 'little', 'image', 'every', 'uncover',
              'are', 'areas', 'based', 'we', 'can', 'changes', 'using', 'be', 'between', 'models', 'which', 'whom',
              'where', 'analysis', 'front', 'write', 'methods', 'method', 'villages', 'images', 'group',
              'what', 'have', 'used', 'were', 'has', 'under', 'lu', 'growth', 'it', 'more', 'these', 'those', 'than',
              'however', 'detection', 'behind', 'various', 'including', 'service', 'variety', 'eco', 'includes',
              'sic', 'forth', 'also', 'uh', 'over', 'precipitation', 'respectively', 'effects', 'showed', 'showing',
              'most', 'detect', 'demand', 'down', 'features', 'lead', 'led', 'place', 'normalization',
              'proposed', 'regional', 'distribution', 'while', 'how', 'who', 'three', 'simulations', 'up', 'our',
              'during', 'degree', 'sharing', 'feature', 'destination', 'services', 'include', 'normal',
              'their', 'its', 'ours', 'remote', 'sensing', 'previous', 'network', 'degrees', 'normalize',
              'series', 'patch', 'generating', 'through', 'such', 'been', 'built', 'article', 'table',
              'difference', 'differences', 'or', 'existing', 'structure', 'buildings', 'potential', 'factors',
              'studies', 'detector', 'times', 'exist', 'identified', 'village', 'assess', 'formation', 'tri',
              'time', 'street', 'show', 'support', 'findings', 'strategies', 'government', 'within', 'mobility',
              'travel', 'analyzing', 'exposure', 'exposed', 'perspective', 'perception', 'perceive', 'percept',
              'human', 'co', 'understanding', 'spat', 'rural', 'activity', 'activities', 'network', 'networks',
              'transport', 'road', 'expose', 'find', 'found', 'may', 'might', 'pan', 'assessment', 'uni', 'bi',
              'transportation', 'location', 'traffic', 'daily', 'among', 'large', 'com', 'geo', 'socio', 'access',
              'scale', 'automatic', 'will', 'inter', 'intra', 'district', 'official', 'form', 'format', 'social',
              'cluster', 'spaces', 'mobile', 'big', 'di', 'trip', 'trips', 'gender', 'genders', 'relate', 'related',
              'sc', 'owe', 'owing', 'hint', 'hinder', 'hidden', 'vital', 'residents', 'streets', 'current', 'rapid',
              'due', 'movement', 'one', 'well']
stopwords2 = ['view images', 'structural racism', 'random forest', 'carried out', 'hong kong', 'new york', 'coordination degree',
              'overall accuracy', 'recent years', 'difference differences', 'human mobility', 'united states', 'mode choice',
              'real world', 'jobs housing', 'remote sensing', 'previous studies', 'con network', 'driving factors', 'not only',
              'time series', 'patch generating', 'have been', 'yang river', 'owing to', 'per cent', 'in which']
stopwords3 = nltk.download('stopwords')
nltk_stopwords3 = set(stopwords.words('english'))
# 2. 加载 BERT(T5L/XLNet) 分词器
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
# tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
# tokenizer = T5Tokenizer.from_pretrained('t5-large')
# tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
stopwords_subwords = nltk_stopwords3.union(set(stopwords1))
# for word in stopwords1+stopwords2:
#     # 将每个屏蔽词转换为子词
#     subwords = tokenizer.tokenize(word)
#     stopwords_subwords.update(subwords)  # 更新子词集合
# 3. 定义分词函数
def bert_tokenize(text):
    """
    使用 BERT 分词器进行分词，并过滤停用词。
    """
    # 使用 BERT 进行子词级分词
    tokens = tokenizer.tokenize(text)
    # tokens = tokenizer.encode(text, )
    # 保留字母组成的词，排除屏蔽词
    filtered_tokens = [token for token in tokens if token.isalpha() and token not in stopwords_subwords]
    return filtered_tokens
def filter_bigrams(bigrams, stop_phrases):
    """
    过滤包含屏蔽词的二元短语
    """
    filtered_bigrams = {phrase: freq for phrase, freq in bigrams.items() if phrase not in stop_phrases}
    return filtered_bigrams
# 4. 定义main函数
def main_proceed():
    # 5. 加载数据
    df = pd.read_csv("sorted/sorted_urban_renewal_OFFICIAL.csv")  # 替换为实际文件路径
    # 6. 指定需要处理的文本列（支持多列）
    columns_to_process = ['abstract']  # 替换为你的列名列表
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
    vectorizer = CountVectorizer(ngram_range=(1, 1))  # 设置 ngram_range 为 (2, 2) 表示只提取二元短语
    X = vectorizer.fit_transform(all_texts)
    # 9. 获取二元短语及其词频
    word_freq = dict(zip(vectorizer.get_feature_names_out(), X.toarray().sum(axis=0)))
    # 过滤包含屏蔽词的二元短语
    filtered_bigrams = filter_bigrams(word_freq, stopwords2)
    # 10. 排序词频
    sorted_word_freq = dict(sorted(filtered_bigrams.items(), key=lambda item: item[1], reverse=True))
    return sorted_word_freq

if __name__=='__main__':
    # 11. 运行main函数
    word_freq_list = main_proceed()
    # 12. 输出前25个最常见的词
    print("前25个最常见的词频：", dict(list(word_freq_list.items())[:30]))
    # 13. 绘制词频图
    rcParams['font.sans-serif'] = ['SimHei', 'Times New Roman']  # 可以根据需要换成其他字体
    rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.figure(figsize=(16, 12))
    # 提取前25个高频二元短语及其频率
    phrases = list(word_freq_list.keys())[:25]
    frequencies = list(word_freq_list.values())[:25]
    # 绘制横向条形图
    plt.barh(phrases, frequencies)  # barh表示横向条形图
    plt.xlabel("频率")
    plt.ylabel("短语/单词")
    plt.title("频率统计")
    plt.gca().invert_yaxis()  # 反转 y 轴，使频率最高的短语在顶部
    # pd.DataFrame(list(word_freq_list.items()),columns=['Phrase', 'Frequency']).to_csv("tables/urban_mobility_patterns_frequency_OFFICIAL.csv",
    #                                                                                   index=False)
    plt.show()
