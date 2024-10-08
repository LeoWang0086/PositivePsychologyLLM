import json

# 假设 'data.json' 是你的 JSON 文件名
file_path = 'PositivePsychologyLLM/data/raw json/人工改编计算机语料.json'



# 读取 JSON 文件
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 遍历每个对话，检查是否有不合格的情况
for dialogue_index, dialogue in enumerate(data):
    conversations = dialogue['conversations']
    
    # 检查对话是否以 "counselor" 开始
    if conversations and conversations[0]['from'] == 'counselor':
        print(f"Dialogue {dialogue_index}: Starts with 'counselor'. This is not valid.")
    
    last_from = None
    
    # 遍历每条对话
    for conv_index, conv in enumerate(conversations):
        current_from = conv['from']

        # 检查是否有连续的相同 'from' 值
        if current_from == last_from:
            print(f"Dialogue {dialogue_index}, Conversation {conv_index}: Consecutive messages from '{current_from}'.")
            print(f"Message: {conv['value']}")
        
        # 更新 last_from 以便下一次比较
        last_from = current_from
