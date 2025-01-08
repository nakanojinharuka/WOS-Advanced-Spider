import pandas as pd
import re


# 提取列名分类和序号
def parse_column(col):
    match = re.match(r"([a-zA-Z]+)(\d+)", col)  # 匹配 "author1", "keyword2" 等格式
    if match:
        prefix, number = match.groups()
        return prefix, int(number)
    return col, float('inf')  # 非匹配列放到最后

# 筛选期刊名称
def filter_row(csv_file_in_path: str, filter_data_txt: str, csv_file_out_str: str):
    # 加载期刊名称参考列表
    with open(f'T1/{filter_data_txt}', 'r', encoding='utf-8') as file:
        reference_journals = [line.strip() for line in file.readlines()]
    # 加载表格数据
    data = pd.read_csv(csv_file_in_path)
    # 期刊名称存储在名为 "journal/book" 的列中
    journal_column = 'journal/book'
    # 改进代码以支持大小写不敏感筛选
    reference_journals_lower = [journal.lower().strip() for journal in reference_journals]
    # 将表格中期刊名称列统一转换为小写后筛选
    filtered_data_case_insensitive = data[data[journal_column].str.lower().str.strip().isin(reference_journals_lower)]
    # 保存改进后的筛选结果到新文件
    output_path_case_insensitive = f'filtered sorted/filtered_{csv_file_out_str}.csv'
    filtered_data_case_insensitive.to_csv(output_path_case_insensitive, index=False)
    return output_path_case_insensitive


if __name__ == "__main__":
    # 读取CSV文件
    file_path = 'raw data/2024_spatial_interaction_urban_OFFICIAL1.csv'  # 替换为你的CSV文件路径
    df = pd.read_csv(file_path)
    # 对列名排序
    sorted_columns = sorted(df.columns, key=lambda x: parse_column(x))
    # 重新排列列顺序
    df = df[sorted_columns]
    # 输出结果到新的CSV文件
    mid_path = 'temp/data/T.csv'
    df.to_csv(mid_path, index=False)
    final_path = filter_row(mid_path, 'Earth Science.txt', '2024_spatial_interaction_urban')
    print(f"Sorted file saved as: {final_path}")
