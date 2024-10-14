import time
import pandas as pd
from zhipuai import ZhipuAI
from tqdm import tqdm  # 用于显示进度条
import os

# 初始化 ZhipuAI 客户端，填写您的 API Key
client = ZhipuAI(api_key="423e6ccbd44c1606030490a9ba849000.XrYNKimn1u2qeERV")

# 从 Excel 文件读取 Therapist 的 prompt
def read_prompt_excel(filepath):
    df = pd.read_excel(filepath)
    intent_detail_list = []
    for index, row in df.iterrows():
        positive_examples = [row['positive example 1'].strip(), row['positive example 2'].strip(), row['positive example 3'].strip()]
        negative_examples = [row['negative example 1'].strip(), row['negative example 2'].strip(), row['negative example 3'].strip()]
        intent_detail_list.append({
            'intent': row['intent'].strip(),
            'definition': row['definition'].strip(),
            'positive_examples': positive_examples,
            'negative_examples': negative_examples
        })
    return intent_detail_list

# 调用 ZhipuAI 获取返回结果
def get_completion_from_messages(messages, temperature=0.7):
    system_message = {
        "role": "system",
        "content": ""
    }
    
    formatted_messages = [system_message] + messages

    time.sleep(1)
    for i in range(3):
        try:
            response = client.chat.completions.create(
                model='glm-4-plus',
                messages=formatted_messages,
                temperature=temperature
            )
            return response.choices[0].message.content  # 正确的属性访问方式
        except Exception as e:
            print(f"Error: {e}, retrying...")
            time.sleep(3 * (2 ** i))
    return ''

# 创建消息的函数，使用多标签方式并带定义和正面例子
def create_message(intent_detail_list, utterance):
    intent_name_list = [intent_detail['intent'] for intent_detail in intent_detail_list]
    intent_name_text = ', '.join(f'"{word}"' for word in intent_name_list)
    
    intent_definition_with_examples_list = []
    for intent_detail in intent_detail_list:
        intent_text = intent_detail['intent']
        definition_text = intent_detail['definition'].replace("\\", "")
        positive_example_list = intent_detail['positive_examples']
        intent_definition_with_examples_list.append(f' {intent_text}: {definition_text} 正面例子: {positive_example_list}')
    
    user_prompt_template = f"请根据这段话判断可能的咨询师意图：{utterance}。\n" \
                           f"意图定义:\n {'; '.join(intent_definition_with_examples_list)}\n" \
                           f"只能从这个列表中选择: [{intent_name_text}]\n" \
                           f"如果找不到答案，请说未知。请你直接告诉我结果，不需要任何其它内容，格式：[intents_list]"
    
    messages = [{'role': 'user', 'content': user_prompt_template}]
    return messages

# 保存数据到 Excel 文件
def save_to_excel(df, filepath, mode='w'):
    # 如果文件存在并且我们想要追加
    if mode == 'a' and os.path.exists(filepath):
        with pd.ExcelWriter(filepath, mode='a', if_sheet_exists='overlay') as writer:
            df.to_excel(writer, index=False, startrow=writer.sheets['Sheet1'].max_row, header=False)
    else:
        df.to_excel(filepath, index=False)
    print(f"Partial results saved to {filepath}")

# 提取意图列表函数，确保返回简洁的结果
def extract_intent_list(response):
    try:
        # 直接提取返回的 intents_list 部分
        start = response.index("[")
        end = response.index("]")
        return response[start:end+1]
    except ValueError:
        # 如果无法找到 intents_list，返回未知
        return "未知"

# 主流程，处理 Excel 中的句子并打印输出结果
def process_excel_file(input_filepath, output_filepath):
    # 读取 Excel 文件的内容
    df = pd.read_excel(input_filepath)

    # 读取 prompt 数据，只需要读取一次
    intent_detail_list = read_prompt_excel('PositivePsychologyLLM/code/intent_classification/prompts/mine-translation of threapist.xlsx')

    # 新增四列，用于保存 ZhipuAI 的返回结果
    df['Generated_Result'] = ''
    df['Ground_Truth_Result'] = ''
    df['Generated_Full_Response'] = ''
    df['Ground_Truth_Full_Response'] = ''

    # 从 Excel 中读取需要处理的句子，并逐行处理
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing", unit=" row"):
        generated_sentence = row['Generated']
        ground_truth_sentence = row['Ground Truth']

        # 对 Generated 句子进行判断
        generated_messages = create_message(intent_detail_list, generated_sentence)
        generated_result = get_completion_from_messages(generated_messages)
        df.at[index, 'Generated_Result'] = extract_intent_list(generated_result)
        df.at[index, 'Generated_Full_Response'] = generated_result  # 保存完整结果

        # 对 Ground Truth 句子进行判断
        ground_truth_messages = create_message(intent_detail_list, ground_truth_sentence)
        ground_truth_result = get_completion_from_messages(ground_truth_messages)
        df.at[index, 'Ground_Truth_Result'] = extract_intent_list(ground_truth_result)
        df.at[index, 'Ground_Truth_Full_Response'] = ground_truth_result  # 保存完整结果

        # 每 10 行保存一次结果
        if (index + 1) % 10 == 0:
            save_to_excel(df.iloc[:index + 1], output_filepath, mode='a')
            print(f"Processed {index + 1} rows.")

    # 保存最终的结果
    save_to_excel(df, output_filepath)
    print(f"Final results saved to {output_filepath}")

# 输入文件和输出文件的路径
input_filepath = 'PositivePsychologyLLM/data/generated data/conversation_results_with_no_and_question-1012.xlsx'
output_filepath = 'PositivePsychologyLLM/data/generated data/conversation_results_with_no_and_question-1012-result with intent.xlsx'

# 处理文件并生成结果
process_excel_file(input_filepath, output_filepath)
