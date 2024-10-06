import os
import gradio as gr
import torch
from threading import Thread
from peft import PeftModel
from typing import Union, Annotated
from pathlib import Path
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer
)

ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

# MODEL_PATH = os.environ.get('MODEL_PATH', '/root/autodl-tmp/chatglm3-6b/')
# TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)

def _resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()



# 新的代码

model_path = '/root/autodl-tmp/LLaMA-Factory/final' # 替换为你导出的模型路径
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map='auto')




class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [0, 2]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text

def get_instruction(dialogue_history):
    instruction = f'''现在你扮演一位专业的积极心理专家，你的名字叫做清小深。你具备丰富的心理学和心理健康知识。你擅长运用多种心理咨询技巧，例如认知行为疗法原则、动机访谈技巧和解决问题导向的短期疗法。以温暖亲切的语气，展现出共情和对来访者感受的深刻理解。以自然的方式与来访者进行对话，避免过长或过短的回应，确保回应流畅且类似人类的对话。提供深层次的指导和洞察，使用具体的心理概念和例子帮助来访者更深入地探索思想和感受。避免教导式的回应，更注重共情和尊重来访者的感受。根据来访者的反馈调整回应，确保回应贴合来访者的情境和需求。请为以下的对话生成一个回复，认清你的角色：
        对话：
        {dialogue_history}'''
    return instruction

def predict(history, max_length, top_p, temperature):
    stop = StopOnTokens()
    messages = []
    for idx, (user_msg, model_msg) in enumerate(history):
        if idx == len(history) - 1 and not model_msg:
            messages.append({"role": "user", "content": user_msg})
            break
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if model_msg:
            messages.append({"role": "assistant", "content": model_msg})

    print("\n\n====conversation====\n", messages)
    dialogue_history = "\n".join([msg["content"] for msg in messages])
    instruction = get_instruction(dialogue_history)
    model_inputs = tokenizer.apply_chat_template([{"role": "user", "content": instruction}],
                                                 add_generation_prompt=True,
                                                 tokenize=True,
                                                 return_tensors="pt").to(next(model.parameters()).device)
    streamer = TextIteratorStreamer(tokenizer, timeout=60, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = {
        "input_ids": model_inputs,
        "streamer": streamer,
        "max_new_tokens": max_length,
        "do_sample": True,
        "top_p": top_p,
        "temperature": temperature,
        "stopping_criteria": StoppingCriteriaList([stop]),
        "repetition_penalty": 1.2,
    }
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    for new_token in streamer:
        if new_token != '':
            history[-1][1] += new_token
            yield history

with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">清小深</h1>""")
    chatbot = gr.Chatbot()

    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10, container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(0, 32768, value=8192, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0.01, 1, value=0.6, step=0.01, label="Temperature", interactive=True)

    def user(query, history):
        return "", history + [[parse_text(query), ""]]

    submitBtn.click(user, [user_input, chatbot], [user_input, chatbot], queue=False).then(
        predict, [chatbot, max_length, top_p, temperature], chatbot
    )
    emptyBtn.click(lambda: None, None, chatbot, queue=False)

demo.queue()
demo.launch(server_name="127.0.0.1", server_port=7870, inbrowser=True, share=False)
