import pandas as pd
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import Dataset
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
def generate_commonality_and_difference_summary_chunked(df, model_name="toloka/t5-large-for-text-aggregation",
                                                        max_token_length=512, batch_size=32):
    """
    从表格中的标题和摘要中提取研究的共性和差异性，以小作文形式总结，支持超长文本分块处理。
    """
    # 提取标题和摘要
    titles = df['title'].dropna().tolist() if 'title' in df.columns else []
    abstracts = df['abstract'].dropna().tolist() if 'abstract' in df.columns else []
    if not titles or not abstracts:
        return "表中缺少 'title' 或 'abstract' 列，无法生成总结。"
    # 整理输入内容
    # 合并标题和摘要为整体文本
    titles_text = " ".join(titles)
    abstracts_text = " ".join(abstracts)
    # 加载模型和分词器
    summarizer = pipeline("summarization", model=model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def process_text(text, intro, batch_size=batch_size):
        """
        对文本分块并批量生成摘要。
        """
        # 分块文本
        tokens = tokenizer(text, truncation=False, return_tensors="pt")["input_ids"][0]
        chunks = [tokenizer.decode(tokens[i:i + max_token_length], skip_special_tokens=True)
                  for i in range(0, len(tokens), max_token_length)]

        # 构造批量输入数据集
        dataset = Dataset.from_dict({"text": chunks})

        def summarize_batch(batch):
            """对每个批次生成摘要"""
            summaries = summarizer(
                [intro + text for text in batch['text']],
                max_length=200,
                min_length=150,
                truncation=True
            )
            return {"summary": [summary['summary_text'] for summary in summaries]}

        # 批量生成摘要
        results = dataset.map(summarize_batch, batched=True, batch_size=batch_size)
        return "\n".join(results["summary"])

    # 对标题和摘要分别生成共性和差异性总结
    commonality_intro = "Across these study, common themes include:"
    difference_intro = "However, key differences in them can be observed in:"

    # 分析
    commonality = process_text(titles_text + abstracts_text, commonality_intro)
    difference = process_text(titles_text + abstracts_text, difference_intro)

    # 整合总结
    result = (
        f"### Commonality Analysis\n\n{commonality}\n\n"
        f"### Difference Analysis\n\n{difference}\n\n"
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
        print("请指定有效的方法，例如commonality_difference")


if __name__ == '__main__': # 示例用法
    file_paths = [f"raw data/2024_mobility_metropolitan_area_OFFICIAL{i}.csv" for i in range(1, 2)]  # 替换为实际的文件路径
    main(file_paths)
