import pandas as pd
import re

# 读取Excel文件，替换成你的Excel文件路径
file_path = 'PositivePsychologyLLM/data/generated data/conversation_results_with_no_and_question-1012-result with intent.xlsx'
df = pd.read_excel(file_path)

# 定义合法的关键词集合
valid_results = {
    "Makes Needs Explicit",
    "Makes Emotions Explicit",
    "Makes Values Explicit",
    "Makes Consequences Explicit",
    "Makes Conflict Explicit",
    "Makes Strengths/Resources Explicit",
    "Evokes Concrete Elaboration",
    "Evokes Perspective Elaboration",
    "Emotions Check-in",
    "Problem-Solving",
    "Planning",
    "Normalizing",
    "Teaching/Psychoeducation"
}

# Mapping 表
mapping_dict = {
    "Makes Needs Explicit": ("needs", "REFLECTIONS ON"),
    "Makes Emotions Explicit": ("emotions", "REFLECTIONS ON"),
    "Makes Values Explicit": ("values", "REFLECTIONS ON"),
    "Makes Consequences Explicit": ("consequences", "REFLECTIONS ON"),
    "Makes Conflict Explicit": ("conflict", "REFLECTIONS ON"),
    "Makes Strengths/Resources Explicit": ("strength", "REFLECTIONS ON"),
    "Evokes Concrete Elaboration": ("experiences", "QUESTIONS ON"),
    "Evokes Perspective Elaboration": ("perspectives", "QUESTIONS ON"),
    "Emotions Check-in": ("emotions", "QUESTIONS ON"),
    "Problem-Solving": ("problem-solving", "SOLUTIONS"),
    "Planning": ("planning", "SOLUTIONS"),
    "Normalizing": ("normalizing", "SOLUTIONS"),
    "Teaching/Psychoeducation": ("psychoeducation", "PSYCHOEDUCATION")
}

# 检查列中每个值是否包含合法的关键词，并提取和删除匹配的子字符串
def extract_and_remove_valid_results(result_column):
    def process_entry(x):
        original_text = str(x).strip().lower()
        extracted = []
        for valid_result in valid_results:
            if valid_result.lower() in original_text:
                extracted.append(valid_result)
                # 删除匹配到的合法关键词
                original_text = original_text.replace(valid_result.lower(), "").strip()
        return extracted, original_text  # 返回提取到的子字符串和剩余的文本

    return result_column.apply(process_entry)

# 应用函数到 Generated_Result 和 Ground_Truth_Result 列
df['Generated_Extracted'], df['Generated_Remaining'] = zip(*extract_and_remove_valid_results(df['Generated_Result']))
df['Ground_Truth_Extracted'], df['Ground_Truth_Remaining'] = zip(*extract_and_remove_valid_results(df['Ground_Truth_Result']))

# 将 intent mapping 应用于 Generated_Extracted 和 Ground_Truth_Extracted
def map_intents(extracted_list):
    intent_1_set = set()
    intent_2_set = set()
    for intent in extracted_list:
        if intent in mapping_dict:
            intent_1, intent_2 = mapping_dict[intent]
            intent_1_set.add(intent_1)
            intent_2_set.add(intent_2)
    return ', '.join(intent_1_set), ', '.join(intent_2_set)  # 使用set去重并返回唯一值

# 应用到生成的提取列上，获取 intent1 和 intent2
df['Generated_Intent1'], df['Generated_Intent2'] = zip(*df['Generated_Extracted'].apply(map_intents))
df['Ground_Truth_Intent1'], df['Ground_Truth_Intent2'] = zip(*df['Ground_Truth_Extracted'].apply(map_intents))

# 保留原始的指定列
selected_columns = [
    'Conversation No', 'Round No', 'Generated', 'Ground Truth', 'History', 'Question', 
    'Generated_Word_Count', 'Ground_Truth_Word_Count', 'rouge-1', 'rouge-2', 
    'rouge-l', 'bleu', 'bert_score'
]

# 选择生成的列
generated_columns = [
    'Generated_Full_Response', 'Ground_Truth_Full_Response', 
    'Generated_Intent1', 'Generated_Intent2', 
    'Ground_Truth_Intent1', 'Ground_Truth_Intent2'
]

# 合并原始列和生成列
final_df = df[selected_columns + generated_columns]

# 定义输出路径
output_file_path = 'PositivePsychologyLLM/data/result/intent_formated/mapped_results_with_selected_columns.xlsx'

# 保存映射后的结果，并保留指定列
final_df.to_excel(output_file_path, index=False)

print(f"Mapped results with selected columns have been saved to {output_file_path}")
