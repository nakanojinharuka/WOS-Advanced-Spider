import pandas as pd
from transformers import BertTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import XLNetTokenizer
from transformers import RobertaTokenizer, RobertaForMaskedLM
import argparse
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib import rcParams
import nltk
from nltk.corpus import stopwords
import glob
import os
# 1. 定义屏蔽词列表（可以自定义或扩展）
stopwords1 = ['the', 'and', 'of', 'in', 'to', 'for', 'a', 'on', 'is', 'with', 'at', 'this', 'an', 's', 'es', 'neo',
              'index', 'people', 'across', 'groups', 'influence', 'central', 'first', 'second', 'third', 'higher', 'ts',
              'urban', 'renewal', 'simulation', 'city', 'project', 'base', 'model', 'study', 'analysis', 'cities', 'pe',
              'services', 'non', 'types', 'levels', 'there', 'object', 'orient', 'oriented', 'horizontal', 'vertical',
              'development', 'redevelopment', 'process', 'public', 'result', 'provide', 'area', 'research', 'policies',
              'into', 'segmentation', 'segment', 'geographical', 'distribute', 'distributed', 'many', 'much', 'item',
              'method', 'building', 'change', 'planning', 'datum', 'data', 'design', 'high', 'low', 'paper', 'making',
              'local', 'small', 'region', 'regions', 'datum', 'locations', 'because', 'cause', 'since', 'when', 'layer',
              'effect', 'case', 'system', 'propose', 'impact', 'strategy', 'new', 'factor', 'level', 'different', 'age',
              'quality', 'network', 'good', 'better', 'best', 'per', 'persist', 'percent', 'items', 'decide', 'im',
              'value', 'approach', 'use', 'utilize', 'increase', 'decrease', 'build', 'space', 'finding', 'old', 'mit',
              'cellular', 'middle', 'bad', 'worse', 'worst', 'complex', 'simple', 'simply', 'very', 'mono', 'waves',
              'young', 'plan', 'significant', 'important', 'framework', 'explore', 'relationship', 'key', 'property',
              'wr', 'medium', 'deserve', 'innovation', 'create', 'creation', 'provides', 'provided', 'properties', 'uc',
              'keywords', 'keyword', 'author', 'plus', 'multi', 'learn', 'information', 'population', 'algorithm', 'sw',
              'auto', 'whose', 'economic', 'gap', 'identify', 'role', 'act', 'effective', 'sheet', 'positive', 'wave',
              'diffusion', 'game', 'social', 'community', 'china', 'policy', 'politic', 'right', 'spatial', 'race', 'ne',
              'impacts', 'efficient', 'media', 'systems', 'play', 'evaluate', 'future', 'everyone', 'negative', 'con',
              'politics', 'participate', 'participation', 'rights', 'state', 'gen', 'post', 'projects', 'land', 'cover',
              'relationships', 'long', 'short', 'res', 'sub', 'take', 'took', 'hydro', 'capita', 'ss', 'ex', 'syn',
              'two', 'four', 'could', 'would', 'should', 'reveal', 'evaluation', 'decision', 'accessible', 'health',
              'neural', 'patterns', 'pattern', 'eva', 'ag', 'learning', 'risk', 'that', 'by', 'from', 'was', 'as', 'po',
              'results', 'analyze', 'frontier', 'less', 'least', 'little', 'image', 'every', 'uncover', 'hat', 'gi',
              'are', 'areas', 'based', 'we', 'can', 'changes', 'using', 'be', 'between', 'models', 'which', 'whom',
              'where', 'analysis', 'front', 'write', 'methods', 'method', 'villages', 'images', 'group', 'het', 'graph',
              'what', 'have', 'used', 'were', 'has', 'under', 'lu', 'growth', 'it', 'more', 'these', 'those', 'than',
              'however', 'detection', 'behind', 'various', 'including', 'service', 'variety', 'eco', 'includes', 'ant',
              'sic', 'forth', 'also', 'uh', 'over', 'precipitation', 'respectively', 'effects', 'showed', 'showing',
              'most', 'detect', 'demand', 'down', 'features', 'lead', 'led', 'place', 'normalization', 'water', 'sp',
              'proposed', 'regional', 'distribution', 'while', 'how', 'who', 'three', 'simulations', 'up', 'our', 'bc',
              'during', 'degree', 'sharing', 'feature', 'destination', 'services', 'include', 'normal', 'interact',
              'their', 'its', 'ours', 'remote', 'sensing', 'previous', 'network', 'degrees', 'normalize', 'interaction',
              'series', 'patch', 'generating', 'through', 'such', 'been', 'built', 'article', 'table', 'interactions',
              'difference', 'differences', 'or', 'existing', 'structure', 'buildings', 'potential', 'factors', 'micro',
              'studies', 'detector', 'times', 'exist', 'identified', 'village', 'assess', 'formation', 'tri', 'macro',
              'time', 'street', 'show', 'support', 'findings', 'strategies', 'government', 'within', 'mobility', 'poll',
              'travel', 'analyzing', 'exposure', 'exposed', 'perspective', 'perception', 'perceive', 'percept', 'air',
              'human', 'co', 'understanding', 'spat', 'rural', 'activity', 'activities', 'network', 'networks', 'river',
              'global', 'bacterial', 'bacteria', 'insight', 'insights', 'surface', 'manage', 'management', 'relation',
              'transport', 'road', 'expose', 'find', 'found', 'may', 'might', 'pan', 'assessment', 'uni', 'bi', 'us',
              'transportation', 'location', 'traffic', 'daily', 'among', 'large', 'com', 'geo', 'socio', 'access', 'ad',
              'scale', 'automatic', 'will', 'inter', 'intra', 'district', 'official', 'form', 'format', 'social', 'tr',
              'cluster', 'spaces', 'mobile', 'big', 'di', 'trip', 'trips', 'gender', 'genders', 'relate', 'related',
              'sc', 'owe', 'owing', 'hint', 'hinder', 'hidden', 'vital', 'residents', 'streets', 'current', 'rapid',
              'due', 'movement', 'one', 'well', 'mechanisms', 'furthermore', 'natural', 'significantly', 'compare',
              'compared', 'comparison', 'comparing', 'multiple', 'often', 'usually', 'seldom', 'always', 'enhance',
              'linear', 'ns', 'smoke', 'fog', 'distance', 'usage', 'st', 'cd', 'bio', 'challenges', 'challenge', 'et',
              'effectively', 'effecting', 'importance', 'older', 'phone', 'conduct', 'conducted', 'conduction', 'imp',
              'pet', 'rest', 'ul', 'deep', 'mapping', 'production', 'nighttime', 'self', 'adapt', 'bridge', 'uv', 'map',
              'industrial', 'functional', 'needs', 'emissions', 'energy', 'improve', 'improvement', 'improving', 'maps',
              'theory', 'control', 'initiatives', 'initiative', 'ethiopian', 'chinese', 'japanese', 'korean', 'life',
              'economy', 'economics', 'optimize', 'optimization', 'optimizing', 'adaptive', 'black', 'crucial', 'entity',
              'entities', 'affect', 'win', 'cycle', 'interview', 'interviews', 'rev', 'six', 'five', 'four', 'un', 'sl',
              'ally', 'evidence', 'stake', 'attach', 'attachment', 'attractive', 'des', 'ballet', 'clearance', 'concrete',
              'deter', 'indirect', 'direct']
stopwords2 = ['view images', 'structural racism', 'random forest', 'carried out', 'hong kong', 'new york', 'coordination degree',
              'overall accuracy', 'recent years', 'difference differences', 'human mobility', 'united states', 'mode choice',
              'real world', 'jobs housing', 'remote sensing', 'previous studies', 'con network', 'driving factors', 'not only',
              'time series', 'patch generating', 'have been', 'yang river', 'owing to', 'per cent', 'in which']
stopwords3 = nltk.download('stopwords')
nltk_stopwords3 = set(stopwords.words('english'))
# 2. 加载 BERT(T5L/XLNet) 分词器，定义各参数
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
# tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
# tokenizer = T5Tokenizer.from_pretrained('t5-large')
# tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
stopwords_subwords = nltk_stopwords3.union(set(stopwords1))
parser = argparse.ArgumentParser()
parser.add_argument('-k', '--file_keyword', default='', required=True, help='keywords for your *.csv files')
parser.add_argument('-g', '--gram_numbers', default=1, required=True, type=int, help='unigram: 1, bi-gram: 2, trigram: 3')
args = parser.parse_args()
# 3. 定义分词函数
def bert_tokenize(text):
    # 使用 BERT 分词器进行分词，并过滤停用词。
    tokens = tokenizer.tokenize(text)
    filtered_tokens = [token for token in tokens if token.isalpha() and token not in stopwords_subwords]
    return filtered_tokens
def filter_bigrams(bigrams, stop_phrases):
    # 过滤包含屏蔽词的二元短语
    filtered_bigrams = {phrase: freq for phrase, freq in bigrams.items() if phrase not in stop_phrases}
    return filtered_bigrams
# 4. 定义main函数
def main_proceed():
    # 5. 加载数据
    ## Path and keyword of the files
    path = r'Filtered_Sorted'
    keyword = args.file_keyword
    gram_numbers = args.gram_numbers
    ## joining the path and creating list of paths
    all_files = [f for f in glob.glob(os.path.join(path, '*.csv')) if keyword in os.path.basename(f).lower()]
    dataframes = []
    ## reading the data and appending the dataframe
    for dfs in all_files:
        data = pd.read_csv(dfs)
        dataframes.append(data)
    ## Concatenating the dataframes
    df = pd.concat(dataframes, ignore_index=True)
    # 6. 指定需要处理的文本列（支持多列）
    columns_to_process = ['abstract', 'title']+[f'keyword{i}' for i in range(1, 30)]
    # 7. 处理多列文本数据
    all_texts = []  # 存储所有列的分词结果
    for col in columns_to_process:
        if col in df.columns:
            # 获取所有文本并去除空值
            texts = df[col].dropna()
            # 应用 BERT 分词
            processed_texts = texts.apply(bert_tokenize).apply(lambda x: ' '.join(x))  # 转成字符串形式
            all_texts.extend(processed_texts)
    # 8. 提取一元、二元、三元短语
    vectorizer = CountVectorizer(ngram_range=(gram_numbers, gram_numbers))
    X = vectorizer.fit_transform(all_texts)
    # 9. 获取短语及其词频
    word_freq = dict(zip(vectorizer.get_feature_names_out(), X.toarray().sum(axis=0)))
    # 过滤屏蔽词
    filtered_bigrams = filter_bigrams(word_freq, stopwords2)
    # 10. 排序
    sorted_word_freq = dict(sorted(filtered_bigrams.items(), key=lambda item: item[1], reverse=True))
    return sorted_word_freq

if __name__=='__main__':
    # 11. 运行main函数
    word_freq_list = main_proceed()
    # 12. 输出前30个最常见的词
    print("前30个最常见的词频：", dict(list(word_freq_list.items())[:30]))
    # 13. 绘制词频图
    # rcParams['font.sans-serif'] = ['SimHei', 'Times New Roman']  # 可以根据需要换成其他字体
    # rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    # plt.figure(figsize=(16, 12))
    # 提取前25个高频短语及其频率
    # phrases = list(word_freq_list.keys())[:25]
    # frequencies = list(word_freq_list.values())[:25]
    # 绘制横向条形图
    # plt.barh(phrases, frequencies)  # barh表示横向条形图
    # plt.xlabel("频率")
    # plt.ylabel("短语/单词")
    # plt.title("频率统计")
    # plt.gca().invert_yaxis()  # 反转 y 轴，使频率最高的短语在顶部
    pd.DataFrame(list(word_freq_list.items()),
                 columns=['Phrase', 'Frequency']).to_csv(f"frequency tables/filtered_{args.file_keyword}_G{args.gram_numbers}.csv",
                                                                                      index=False)
    # plt.show()
