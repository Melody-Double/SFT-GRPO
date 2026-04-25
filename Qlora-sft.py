import warnings
import logging
import os
import shutil
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
if os.path.exists("unsloth_compiled_cache"):
    shutil.rmtree("unsloth_compiled_cache")
    print("�� 已经清理旧的 Unsloth 缓存！")
os.environ['UNSLOTH_USE_MODELSCOPE'] = '1'
os.environ['UNSLOTH_SKIP_STATISTICS'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['UNSLOTH_COMPILE_DISABLE'] = '1'
os.environ['UNSLOTH_DISABLE_FAST_GENERATION'] = '1'
os.environ['UNSLOTH_DISABLE_AUTO_UPDATES'] = '1'


from unsloth import FastLanguageModel, is_bfloat16_supported, unsloth_train
from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/home/amax/double/sft/SFT-RLHF/model/DeepSeek-R1-Distill-Qwen-1.5B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    local_files_only= True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
SYSTEM_PROMPT = "你是一位医学专家，具备临床推理、疾病诊断和治疗方案制定方面的专业能力。请回答下面的医学问题。"
GENERATION_PREFIX = "<think>\n"
EOS_TOKEN = tokenizer.eos_token  # 一定要添加结束标记，否则会无限生成

def formatting_prompts_func(examples):
    inputs = examples["Question"]
    cots = examples["Complex_CoT"]
    outputs = examples["Response"]
    texts = []
    for input, cot, output in zip(inputs, cots, outputs):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input},
            {
                "role": "assistant",
                "content": f"<think>\n{cot}\n</think>\n<answer>\n{output}\n</answer>",
            },
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        ) + EOS_TOKEN
        texts.append(text)
    return {
        "text": texts,
    }

from datasets import load_dataset
dataset = load_dataset(
    "/home/amax/double/sft/SFT-RLHF/dataset/medical-o1-reasoning-sft",
    "zh",
    split="train"
)

print(dataset.column_names)
dataset = dataset.map(formatting_prompts_func, batched = True,remove_columns=dataset.column_names)
print(dataset["text"][0])
FastLanguageModel.for_training(model)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,#每个设备训练批次大小
        gradient_accumulation_steps = 4,#梯度积累步数
        warmup_steps = 5,#预热步数
        # max_steps = 60,#最大训练步数
        num_train_epochs = 1, # For longer training runs!
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",#优化器
        weight_decay = 0.01,
        lr_scheduler_type = "linear",#学习率调度器类型。这里选择 linear 表示学习率将会线性地从初始学习率降低到0。可以帮助模型逐步收敛。
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
        average_tokens_across_devices = False
    ),
)

trainer_stats = trainer.train()
print(trainer_stats)
model.save_pretrained("DeepSeek-R1-Distill-Qwen-1.5B-sft")
tokenizer.save_pretrained("DeepSeek-R1-Distill-Qwen-1.5B-sft")
print("�� SFT 模型已成功保存到 DeepSeek-R1-Distill-Qwen-1.5B-sft 文件夹！")

question = "一个患有急性阑尾炎的病人已经发病5天，腹痛稍有减轻但仍然发热，在体检时发现右下腹有压痛的包块，此时应如何处理？"
print(question)

FastLanguageModel.for_inference(model)  # Unsloth has 2x faster inference!
inference_messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": question},
]
prompt_text = tokenizer.apply_chat_template(
    inference_messages,
    tokenize=False,
    add_generation_prompt=True,
) + GENERATION_PREFIX
inputs = tokenizer([prompt_text], return_tensors="pt").to("cuda")

outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=1200,
    use_cache=True,
)
response = tokenizer.batch_decode(outputs)
print(response[0].split(GENERATION_PREFIX, 1)[-1])
