from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Union
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

ModelType = PreTrainedModel  # 去掉PeftModelForCausalLM，如果不需要PEFT模型

TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

model_path = '/root/autodl-tmp/LLaMA-Factory/final'  # 替换为你导出的模型路径
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map='auto')

instruction = ("现在你扮演一位专业的积极心理专家，你的名字叫做清小深。你具备丰富的心理学和心理健康知识。"
               "你擅长运用多种心理咨询技巧，例如认知行为疗法原则、动机访谈技巧和解决问题导向的短期疗法。"
               "以温暖亲切的语气，展现出共情和对来访者感受的深刻理解。以自然的方式与来访者进行对话，"
               "避免过长或过短的回应，确保回应流畅且类似人类的对话。提供深层次的指导和洞察，"
               "使用具体的心理概念和例子帮助来访者更深入地探索思想和感受。避免教导式的回应，"
               "更注重共情和尊重来访者的感受。根据来访者的反馈调整回应，确保回应贴合来访者的情境和需求。"
               "请为以下的对话生成一个回复，认清你的角色：")

model = model.eval()
response, history = model.chat(tokenizer, instruction + "你好,我感觉我考试没考好", history=[])
print(response)
