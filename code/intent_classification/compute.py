import pandas as pd

# 从文件读取已经映射完成的结果
file_path = 'PositivePsychologyLLM/data/result/intent_formated/mapped_results_with_selected_columns.xlsx'

df = pd.read_excel(file_path)

# 计算重复度的函数
def calculate_overlap_ratio(generated_intent, ground_truth_intent):
    # 将intent字符串转换为集合
    generated_set = set(str(generated_intent).split(', '))
    ground_truth_set = set(str(ground_truth_intent).split(', '))
    
    if not ground_truth_set:  # 如果Ground Truth为空，直接返回0
        return 0
    
    # 计算交集的大小与 Ground Truth 的比例
    overlap_count = len(generated_set & ground_truth_set)  # 交集
    total_count = len(ground_truth_set)  # Ground Truth 中的元素数
    
    return overlap_count / total_count

# 分别计算 Intent1 和 Intent2 的重复度
df['Intent1_Overlap_Ratio'] = df.apply(lambda row: calculate_overlap_ratio(row['Generated_Intent1'], row['Ground_Truth_Intent1']), axis=1)
df['Intent2_Overlap_Ratio'] = df.apply(lambda row: calculate_overlap_ratio(row['Generated_Intent2'], row['Ground_Truth_Intent2']), axis=1)

# 计算整体的平均值
overall_mean_intent1 = df['Intent1_Overlap_Ratio'].mean()
overall_mean_intent2 = df['Intent2_Overlap_Ratio'].mean()

# 按 Round No 分组并计算平均值
grouped_means = df.groupby('Round No')[['Intent1_Overlap_Ratio', 'Intent2_Overlap_Ratio']].mean()

# 打印整体的平均值
print(f"Overall Intent1 Overlap Ratio Mean: {overall_mean_intent1:.4f}")
print(f"Overall Intent2 Overlap Ratio Mean: {overall_mean_intent2:.4f}")

# 打印 groupby Round No 的平均值
print("\nGrouped by Round No - Mean Overlap Ratios:")
print(grouped_means)
