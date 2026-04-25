import warnings
import logging
import os
import shutil
import json
import re
import networkx as nx
import jieba
import torch

# 1. 基础环境清理与配置 (防 Bug 三板斧)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)

if os.path.exists("unsloth_compiled_cache"):
    shutil.rmtree("unsloth_compiled_cache")
    print("�� 已清理旧的 Unsloth 缓存！")

os.environ['UNSLOTH_SKIP_STATISTICS'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['UNSLOTH_COMPILE_DISABLE'] = '1'
os.environ['UNSLOTH_DISABLE_FAST_GENERATION'] = '1'

from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig

# ==========================================
# 2. 构建内存级医疗知识图谱
# ==========================================
print("�� 正在加载三元组数据并构建知识图谱...")
kg_graph = nx.Graph()

# 请确保这里的路径正确指向你抽取的 json
with open("/home/amax/double/sft/re/my_dataset_triples.json", "r", encoding="utf-8") as f:
    kg_data = json.load(f)

for item in kg_data:
    for spo in item.get("spo_list", []):
        if len(spo) == 3:
            s, p, o = spo
            weight = 0.0 if p in ["同义词", "别名", "成份"] else 1.0
            kg_graph.add_edge(s, o, relation=p, weight=weight)

for node in kg_graph.nodes():
    jieba.add_word(node)

print(f"✅ 图谱构建完成！共包含 {kg_graph.number_of_nodes()} 个医学概念节点。")


# ==========================================
# 3. 终极 GRPO 奖励函数设计 (PRM + Penalty + Soft Format)
# ==========================================
def extract_answer(text):
    """提取最终输出的 <answer> 内容"""
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    return match.group(1).strip() if match else ""


# �� 奖励 1：阶梯式平滑格式奖励 (Soft Format Reward)
def soft_format_reward_func(completions, **kwargs):
    rewards = []
    for comp in completions:
        text = comp[0]['content'] if isinstance(comp, list) else comp
        score = 0.0
        if "<think>" in text: score += 0.25
        if "</think>" in text: score += 0.25
        if "<answer>" in text: score += 0.25
        if "</answer>" in text: score += 0.25
        rewards.append(score)
    return rewards


# �� 奖励 2：过程奖励模型 (PRM) - 奖励优质的临床推导过程
def prm_reward_func(completions, ground_truth, **kwargs):
    rewards = []
    for comp, gt in zip(completions, ground_truth):
        text = comp[0]['content'] if isinstance(comp, list) else comp

        # 提取 <think> 里面的推理过程
        # 把严格匹配 </think> 改成：抓取 <think> 之后的所有内容，直到遇到 <answer>
        match = re.search(r'<think>(.*?)(?:</think>|<answer>|$)', text, re.DOTALL)
        think_content = match.group(1) if match else ""

        if not think_content:
            rewards.append(0.0)
            continue

        gt_entities = [w for w in jieba.lcut(gt) if w in kg_graph.nodes]
        think_entities = [w for w in jieba.lcut(think_content) if w in kg_graph.nodes]

        if not gt_entities or not think_entities:
            rewards.append(0.0)
            continue

        # 如果推理过程中提到了图谱里的相关节点(距离<=2)，给予 +0.5 过程推理奖励
        bonus = 0.0
        for t_ent in think_entities:
            for g_ent in gt_entities:
                if t_ent == g_ent:
                    bonus = 0.5
                    break
                try:
                    d = nx.shortest_path_length(kg_graph, t_ent, g_ent, weight='weight')
                    if d <= 2.0:
                        bonus = 0.5
                        break
                except nx.NetworkXNoPath:
                    pass
            if bonus > 0: break

        rewards.append(bonus)
    return rewards


# �� 奖励 3：KG 结果对齐与严厉惩罚 (Outcome & Fatal Penalty)
def kg_and_penalty_reward_func(completions, ground_truth, **kwargs):
    rewards = []
    for comp, gt in zip(completions, ground_truth):
        text = comp[0]['content'] if isinstance(comp, list) else comp
        pred_ans = extract_answer(text)

        # 致命错误 1：完全没写答案
        if not pred_ans:
            rewards.append(-2.0)
            continue

        # 话痨惩罚：要求答案精简，废话超过 50 个字扣 0.5 分
        length_penalty = -0.5 if len(pred_ans) > 50 else 0.0

        gt_entities = [w for w in jieba.lcut(gt) if w in kg_graph.nodes]
        pred_entities = [w for w in jieba.lcut(pred_ans) if w in kg_graph.nodes]

        if not gt_entities:
            score = 1.5 if gt in pred_ans else -1.0
            rewards.append(score + length_penalty)
            continue

        # 致命错误 2：答非所问，找不出任何医学实体
        if not pred_entities:
            rewards.append(-2.0)
            continue

        min_dist = 999.0
        for p_ent in pred_entities:
            for g_ent in gt_entities:
                if p_ent == g_ent:
                    min_dist = 0.0
                    break
                try:
                    d = nx.shortest_path_length(kg_graph, p_ent, g_ent, weight='weight')
                    min_dist = min(min_dist, d)
                except nx.NetworkXNoPath:
                    pass

        # 基于图谱距离的打分与致命错误重罚
        if min_dist == 0.0:
            score = 2.0  # 完美命中
        elif min_dist <= 1.0:
            score = 1.0  # 强相关
        elif min_dist <= 2.0:
            score = 0.5  # 弱相关
        else:
            score = -2.0  # 致命错误：胡乱诊断，直接重罚 -2.0

        rewards.append(score + length_penalty)

    return rewards


# ==========================================
# 4. 加载 SFT 模型与结构化 Prompt 注入
# ==========================================
model_path = "/home/amax/double/sft/SFT-RLHF/DeepSeek-R1-Distill-Qwen-1.5B-sft"
max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    local_files_only=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# �� 核心优化：注入临床思维框架
SYSTEM_PROMPT = """你是一位资深的主治医师。请严格按照以下步骤思考：
1. 提取患者的核心症状与体征。
2. 进行鉴别诊断，排除不可能的疾病。
3. 得出唯一确定的最终诊断或治疗方案。
思考过程必须写在 <think> 标签内，最终的核心结论必须写在 <answer> 标签内，且答案要求精简。"""
GENERATION_PREFIX = "<think>\n"


def prep_grpo_data(examples):
    prompts = []
    for q in examples["Question"]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        ) + GENERATION_PREFIX
        prompts.append(prompt)
    return {"prompt": prompts, "ground_truth": examples["Response"]}


dataset = load_dataset("/home/amax/double/sft/dataset/medical-o1-reasoning-sft", "zh", split="train[0:1000]")
grpo_dataset = dataset.map(prep_grpo_data, batched=True)

# ==========================================
# 5. 超参数调优与启动 (Hyperparameter Tuning)
# ==========================================
training_args = GRPOConfig(
    output_dir="/home/amax/double/sft/outputs_grpo_v3",
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_generations=4,  # 如果你的 4090 显存够，这里可以改成 8，效果会更好
    max_prompt_length=512,
    max_completion_length=1024,  # �� 放宽生成长度，防止 <answer> 被截断
    beta=0.1,  # �� 拉紧 KL 散度，防止模型胡言乱语作弊
    temperature=0.8,  # �� 增加生成多样性，让模型尝试不同推理路径
    logging_steps=1,
    max_steps=100,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    average_tokens_across_devices=False
)

FastLanguageModel.for_training(model)

# 注册三大奖励函数
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[soft_format_reward_func, prm_reward_func, kg_and_penalty_reward_func],
    args=training_args,
    train_dataset=grpo_dataset,
)

# 修复 Unsloth 多模态兼容 Bug
trainer.image_token_id = None
trainer.vision_start_token_id = None
trainer.vision_end_token_id = None

print("�� 开始进阶版 GRPO 强化学习炼丹...")
trainer_stats = trainer.train()

model.save_pretrained("DeepSeek-R1-Distill-Qwen-1.5B-sft-rlhf")
tokenizer.save_pretrained("DeepSeek-R1-Distill-Qwen-1.5B-sft-rlhf")
print("�� 包含全链路推导能力的医学大模型 v2 保存成功！")
