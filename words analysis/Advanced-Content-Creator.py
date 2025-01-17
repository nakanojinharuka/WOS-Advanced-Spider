import pandas as pd
from transformers import pipeline, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. 加载数据
def load_data(file_paths):
    """加载多张表并合并"""
    dataframes = [pd.read_csv(file) for file in file_paths]
    return pd.concat(dataframes, ignore_index=True)

# 2. 文本预处理
def preprocess_texts(df):
    """提取摘要、标题和关键词列并合并"""
    columns_to_process = ['abstract', 'title'] + [f'keyword{i}' for i in range(1, 30)]
    all_texts = []
    for col in columns_to_process:
        if col in df.columns:
            all_texts.extend(df[col].dropna().astype(str).tolist())
    return all_texts

# 3. 基于 TF-IDF 的关键词提取
def extract_keywords_with_tfidf(texts, top_n=10):
    """基于 TF-IDF 提取关键词"""
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.sum(axis=0).A1
    keyword_scores = dict(zip(feature_names, scores))
    sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
    return [keyword for keyword, _ in sorted_keywords[:top_n]]

# 4. 基于 pipeline 的关键词提取
def extract_keywords_with_pipeline(texts, top_n=10):
    """使用 transformers 的 pipeline 提取关键词"""
    fill_mask = pipeline("fill-mask", model="bert-base-uncased")
    keywords = []
    for text in texts:
        if len(text.split()) > 0:
            masked_text = text.split()[0] + " [MASK] " + " ".join(text.split()[1:])
            predictions = fill_mask(masked_text)
            keywords.extend([pred["token_str"] for pred in predictions[:top_n]])
    return list(set(keywords))

# 5. 使用 T5 的总结生成方法（短总结）
def generate_summary_with_t5(texts, model_name="t5-base"):
    """使用 T5 模型生成总结"""
    summarizer = pipeline("summarization", model=model_name)
    combined_text = " ".join(texts)
    max_input_length = 512
    if len(combined_text.split()) > max_input_length:
        combined_text = " ".join(combined_text.split()[:max_input_length])
    summary = summarizer(combined_text, max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# 6. 使用 BART 的长总结生成方法（小文章）
def generate_article_summary(texts, model_name="facebook/bart-large-cnn"):
    """生成 3-4 段，总共 2000-3000 单词的小文章总结"""
    summarizer = pipeline("summarization", model=model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    combined_text = " ".join(texts)
    max_token_length = 512
    target_word_count = 4000
    max_summary_length = 800
    min_summary_length = 400
    tokens = tokenizer(combined_text, truncation=True, return_tensors="pt", max_length=512)["input_ids"][0]
    chunks = [tokens[i:i+max_token_length] for i in range(0, len(tokens), max_token_length)]
    summaries = []
    current_word_count = 0
    for chunk in chunks:
        if current_word_count >= target_word_count:
            break
        text_chunk = tokenizer.decode(chunk, skip_special_tokens=True)
        summary = summarizer(text_chunk, max_length=max_summary_length, min_length=min_summary_length, do_sample=False)
        summary_text = summary[0]['summary_text']
        summaries.append(summary_text)
        current_word_count += len(summary_text.split())
    return "\n\n".join(summaries)


# 7. 提取共性与差异性，生成小作文
def split_text_into_chunks(text, max_token_length, tokenizer):
    """
    将长文本分割为多个块，每块长度不超过 max_token_length。
    """
    tokens = tokenizer(text, truncation=False, return_tensors="pt", max_length=1024)["input_ids"][0]
    chunks = [tokens[i:i + max_token_length] for i in range(0, len(tokens), max_token_length)]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]


def generate_commonality_and_difference_summary_chunked(df, model_name="t5-base",
                                                        max_token_length=1024):
    """
    从表格中的标题和摘要中提取研究的共性和差异性，以小作文形式总结，支持超长文本分块处理。
    """
    # 提取标题和摘要
    titles = df['title'].dropna().tolist() if 'title' in df.columns else []
    abstracts = df['abstract'].dropna().tolist() if 'abstract' in df.columns else []
    if not titles or not abstracts:
        return "表中缺少 'title' 或 'abstract' 列，无法生成总结。"
    # 整理输入内容
    combined_texts = titles + abstracts
    combined_text = " ".join(combined_texts)
    # 加载模型和分词器
    summarizer = pipeline("summarization", model=model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 分块处理长文本
    text_chunks = split_text_into_chunks(combined_text, max_token_length, tokenizer)
    # 提炼共性和差异性
    commonality_intro = "Across these studies, common themes include:"
    difference_intro = "However, key differences can be observed in:"
    commonality_summaries = []
    difference_summaries = []
    for chunk in text_chunks:
        commonality_summary = summarizer(
            commonality_intro + chunk,
            max_length=800,
            min_length=600,
            do_sample=False
        )[0]['summary_text']
        difference_summary = summarizer(
            difference_intro + chunk,
            max_length=800,
            min_length=600,
            do_sample=False
        )[0]['summary_text']
        commonality_summaries.append(commonality_summary)
        difference_summaries.append(difference_summary)
    # 合并总结
    result = (
            "### Commonality Analysis\n\n"
            + "\n".join(commonality_summaries)
            + "\n\n### Difference Analysis\n\n"
            + "\n".join(difference_summaries)
    )
    return result


# 8. 主流程
def main(file_paths, method="commonality_difference"):
    """主程序，支持不同的摘要和关键词提取方式"""
    df = load_data(file_paths)
    if method == "commonality_difference":
        analysis_summary = generate_commonality_and_difference_summary_chunked(df)
        print("\n生成的共性与差异性小作文:")
        print(analysis_summary)
    else:
        print("请指定有效的方法，例如 'commonality_difference'。")


if __name__ == '__main__': # 示例用法
    file_paths = [f"raw data/2024_spatial_interaction_urban_OFFICIAL{i}.csv" for i in range(1, 3)]  # 替换为实际的文件路径
    main(file_paths)
