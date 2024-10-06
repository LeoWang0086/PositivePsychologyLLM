# 把历史记录放到history里面
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载模型和tokenizer
model_path = '/root/autodl-tmp/LLaMA-Factory/final'  # 替换为你导出的模型路径
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map='auto')

# 加载对话数据
with open('test-conversations.json', 'r') as f:
    data = json.load(f)

# 存储对话结果的列表
results = []

# 新的 instruction
instruction_base = (
    "现在你扮演一位专业的积极心理专家，你的名字叫做清小深。你具备丰富的心理学和心理健康知识。"
    "你擅长运用多种心理咨询技巧，例如认知行为疗法原则、动机访谈技巧和解决问题导向的短期疗法。"
    "以温暖亲切的语气，展现出共情和对来访者感受的深刻理解。以自然的方式与来访者进行对话，"
    "避免过长或过短的回应，确保回应流畅且类似人类的对话。提供深层次的指导和洞察，"
    "使用具体的心理概念和例子帮助来访者更深入地探索思想和感受。避免教导式的回应，"
    "更注重共情和尊重来访者的感受。根据来访者的反馈调整回应，确保回应贴合来访者的情境和需求。"
)

# 遍历每个对话
for conversation_no, conversation in enumerate(data, start=1):
    round_no = 1  # 记录每个对话中的轮次
    
    # 用于存储对话历史的列表
    history_for_model = []
    
    for i, turn in enumerate(conversation['conversations']):
        if turn['from'] == 'client':
            # 客户的消息加入 history_for_model，映射 "client" 为 "user"
            history_for_model.append({
                "role": "user",
                "content": turn['value']
            })
        
        elif turn['from'] == 'counselor':
            # 当前问题是客户的最后一条消息
            current_question = history_for_model[-1]['content']  # client 的最后一条消息
            
            # 拼接历史对话（去掉当前轮问题）作为输入
            previous_history = ' + '.join([f"{h['role']} : {h['content']}" for h in history_for_model[:-1]])
            
            # 拼接 instruction_base 和当前的客户问题
            instruction = instruction_base + f"\nclient: {current_question}"
            
            # 生成回复，使用模型的 `chat` 方法，传入前 i-1 轮的历史对话
            try:
                response, updated_history = model.chat(tokenizer, instruction, history=history_for_model[:-1])
            except ValueError as e:
                print(f"Error encountered: {e}")
                print(f"Response content: {response}")
                continue  # 跳过当前轮次，继续处理下一轮
            
            # 处理返回内容，确保格式正确
            response_parts = response.split("\n", maxsplit=1)
            if len(response_parts) == 2:
                metadata, content = response_parts
            else:
                content = response_parts[0]  # 如果没有 metadata，只使用内容部分
            
            # 获取 ground truth
            ground_truth = turn['value']
            
            # 保存结果到字典
            result = {
                "Conversation No": conversation_no,
                "Round No": round_no,
                "Generated": content,
                "Ground Truth": ground_truth,
                "History": previous_history if previous_history else "none",  # 第一轮时没有历史对话
                "Question": current_question  # 当前问题
            }
            
            # 将结果添加到列表
            results.append(result)
            
            # 打印生成的回复和 ground truth
            print(f"Generated: {content}")
            print(f"Ground Truth: {ground_truth}")
            
            # 更新 history_for_model，加入 counselor 的真实对话内容（从 JSON 中读取）
            history_for_model.append({
                "role": "assistant",
                "content": turn['value']  # 从 JSON 文件中获取的真实对话
            })
            
            # 更新轮次
            round_no += 1

# 将结果保存为 pandas DataFrame，并写入 Excel
df = pd.DataFrame(results)
df.to_excel('conversation_results_with_no_and_question.xlsx', index=False)

print("对话结果已成功保存到 conversation_results_with_no_and_questio3.xlsx")
