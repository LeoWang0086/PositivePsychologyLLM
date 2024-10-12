import pandas as pd
import jieba
from rouge import Rouge
from sacrebleu import sentence_bleu
from bert_score import score
import torch

# 定义函数计算 ROUGE 分数
def calculate_rouge_scores(row):
    if not row['Generated'] or not row['Ground Truth']:
        return pd.Series({
            'rouge-1': None,
            'rouge-2': None,
            'rouge-l': None
        })

    output_seg = " ".join(jieba.cut(str(row['Generated'])))
    ground_truth_seg = " ".join(jieba.cut(str(row['Ground Truth'])))

    rouge = Rouge()
    scores = rouge.get_scores(output_seg, ground_truth_seg)
    return pd.Series({
        'rouge-1': scores[0]['rouge-1']['f'],
        'rouge-2': scores[0]['rouge-2']['f'],
        'rouge-l': scores[0]['rouge-l']['f']
    })

# 使用 sacrebleu 计算 BLEU 分数
def calculate_bleu_score(row):
    if not row['Generated'] or not row['Ground Truth']:
        return None

    output_words = " ".join(jieba.cut(str(row['Generated'])))
    ground_truth_words = " ".join(jieba.cut(str(row['Ground Truth'])))

    bleu_score = sentence_bleu(output_words, [ground_truth_words]).score
    return bleu_score

# 定义函数计算 BERTScore
def calculate_bert_score(row):
    if not row['Generated'] or not row['Ground Truth']:
        return None

    P, R, F1 = score([str(row['Generated'])], [str(row['Ground Truth'])], lang='zh', rescale_with_baseline=True)
    return F1.item()

# 定义函数计算字符串的字数
def count_words(text):
    if not text:
        return 0
    return len(jieba.lcut(text))

# 读取 Excel 文件
df = pd.read_excel('./PositivePsychologyLLM/data/generated data/conversation_results_with_no_and_question-1012.xlsx')

# 处理缺失值，将 NaN 值替换为空字符串
df['Generated'] = df['Generated'].fillna('')
df['Ground Truth'] = df['Ground Truth'].fillna('')

# 计算 Generated 和 Ground Truth 的字数
df['Generated_Word_Count'] = df['Generated'].apply(count_words)
df['Ground_Truth_Word_Count'] = df['Ground Truth'].apply(count_words)

# 应用函数计算各个指标并按行添加
df[['rouge-1', 'rouge-2', 'rouge-l']] = df.apply(calculate_rouge_scores, axis=1)
df['bleu'] = df.apply(calculate_bleu_score, axis=1)
df['bert_score'] = df.apply(calculate_bert_score, axis=1)

# 保存到 Excel 文件中
df.to_excel('./PositivePsychologyLLM/data/generated data/conversation_results_with_no_and_question-1012.xlsx', index=False)

# 计算平均值时忽略 None 值
average_rouge_1 = df['rouge-1'].dropna().mean()
average_rouge_2 = df['rouge-2'].dropna().mean()
average_rouge_l = df['rouge-l'].dropna().mean()
average_bleu = df['bleu'].dropna().mean()
average_bert_score = df['bert_score'].dropna().mean()
average_generated_word_count = df['Generated_Word_Count'].mean()
average_ground_truth_word_count = df['Ground_Truth_Word_Count'].mean()

# 打印平均值
print(f"Average ROUGE-1: {average_rouge_1:.4f}")
print(f"Average ROUGE-2: {average_rouge_2:.4f}")
print(f"Average ROUGE-L: {average_rouge_l:.4f}")
print(f"Average BLEU: {average_bleu:.4f}")
print(f"Average BERTScore: {average_bert_score:.4f}")
print(f"Average Generated Word Count: {average_generated_word_count:.2f}")
print(f"Average Ground Truth Word Count: {average_ground_truth_word_count:.2f}")
