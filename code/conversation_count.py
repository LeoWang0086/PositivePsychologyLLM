import json

# 打开并读取 JSON 文件
with open('test-conversations.json', 'r') as f:
    data = json.load(f)

# 统计 JSON 中的 conversation 数量
conversation_count = len(data)

print(f"Total number of conversations: {conversation_count}")
