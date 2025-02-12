import pandas as pd
import re
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-ft", "--file_topic", required=True, type=str, default="2024_spatial_interaction_urban")
parser.add_argument("-fn", "--number_of_files", required=True, type=int, default=1)
parser.add_argument("-yr", "--year_of_publish", required=True, type=str, default="2025")
parser.add_argument("-tj", "--tier_of_journal", type=int, default=1)
parser.add_argument("-if", "--input_folder", type=str, default="raw data")
parser.add_argument("-of", "--output_folder", type=str, default="Filtered_Sorted")
args = parser.parse_args()
reference_journals = []

# 提取列名分类和序号
def parse_column(col):
    match = re.match(r"([a-zA-Z]+)(\d+)", col)  # 匹配 "author1", "keyword2" 等格式
    if match:
        prefix, number = match.groups()
        return prefix, int(number)
    return col, float('inf')  # 非匹配列放到最后

# 筛选期刊名称
def filter_row(csv_file_in_path: str, filter_data_txt_list: list[str], csv_file_out_str: str):
    # 加载期刊名称参考列表
    for filter_data_txt in filter_data_txt_list:
        with open(f'T{args.tier_of_journal}/{filter_data_txt}', 'r', encoding='utf-8') as file:
            reference_journals.append([line.strip() for line in file.readlines()])
    references = []
    for reference_journal_ls in reference_journals:
        for reference_journal in reference_journal_ls:
            references.append(reference_journal)
    # 加载表格数据
    data = pd.read_csv(csv_file_in_path)
    # 期刊名称存储在名为 "journal/book" 的列中
    journal_column = 'journal/book'
    # 改进代码以支持大小写不敏感筛选
    reference_journals_lower = [journal.lower().strip() for journal in references]
    # 将表格中期刊名称列统一转换为小写后筛选
    filtered_data_case_insensitive = data[data[journal_column].str.lower().str.strip().isin(reference_journals_lower)]
    # 保存改进后的筛选结果到新文件
    output_path_case_insensitive = f'{args.output_folder}/filtered_{csv_file_out_str}.csv'
    filtered_data_case_insensitive.to_csv(output_path_case_insensitive, index=False)
    return output_path_case_insensitive


if __name__ == "__main__":
    for file in [f'{args.input_folder}/{args.file_topic}_OFFICIAL{i+1}.csv' for i in range(0, args.number_of_files)]: # 替换为你的CSV文件路径
        # 读取CSV文件
        df = pd.read_csv(file)
        # 对列名排序
        sorted_columns = sorted(df.columns, key=lambda x: parse_column(x))
        # 重新排列列顺序
        df = df[sorted_columns]
        # 输出结果到新的CSV文件
        mid_path = 'temp/data/T.csv'
        df.to_csv(mid_path, index=False)
        final_path = filter_row(mid_path,
                                [
                                    'Earth Science.txt',
                                    'Economics.txt',
                                    'Environmental and Ecological Science.txt',
                                    'Computer Science.txt',
                                ],
                                f'{args.file_topic}{args.number_of_files}')
        print(f"Sorted file saved as: {final_path}")
