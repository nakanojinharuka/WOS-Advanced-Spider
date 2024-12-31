import pandas as pd
import re
# 提取列名分类和序号
def parse_column(col):
    match = re.match(r"([a-zA-Z]+)(\d+)", col)  # 匹配 "author1", "keyword2" 等格式
    if match:
        prefix, number = match.groups()
        return prefix, int(number)
    return col, float('inf')  # 非匹配列放到最后

if __name__ == "__main__":
    # 读取CSV文件
    file_path = 'urban_land_simulation.csv'  # 替换为你的CSV文件路径
    df = pd.read_csv(file_path)
    # 对列名排序
    sorted_columns = sorted(df.columns, key=lambda x: parse_column(x))
    # 重新排列列顺序
    df = df[sorted_columns]
    # 输出结果到新的CSV文件
    output_path = 'sorted/sorted_urban_land_simulation.csv'
    df.to_csv(output_path, index=False)
    print(f"Sorted file saved as: {output_path}")
