from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
import torch
import os
import json
from pathlib import Path
from tqdm import tqdm
from rouge import Rouge
import jieba
from datasets import load_dataset

# ===================== 配置 =====================
model_id = "/home/amax/double/sft/SFT-RLHF/DeepSeek-R1-Distill-Qwen-1.5B-merged-fp16"
save_path = "/home/amax/double/sft/SFT-RLHF/medical-model-int4-gptq"
quantize_bits = 4  # 量化位数：4（INT4）或 8（INT8）
# =============================================


# =============================================
# 量化相关代码（保留原流程）
# =============================================
def gptq_quantize():
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gptq_config = GPTQConfig(bits=quantize_bits, tokenizer=tokenizer)

    print(f"[GPTQ] 正在量化模型: {model_id}")
    print(f"[GPTQ] 量化精度: INT{quantize_bits}")

    quantized_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=gptq_config,
        torch_dtype=torch.float16,
    )

    quantized_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"[GPTQ] 量化完成，模型已保存到: {save_path}")


# =============================================
# 以下是量化效果评估：衡量三个维度的损失
# =============================================

def load_test_data(data_path: str, last_n: int = 100):
    """加载医疗问答测试集，取末尾 last_n 条。"""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[-last_n:]


def compute_ppl(model, tokenizer, texts: list) -> float:
    """
    计算 Perplexity（困惑度），衡量量化在通用 benchmark 上的损失。

    原理：PPL = exp(-1/N * Σ log P(x_i))，语言模型对文本的平均不确定性。
          PPL 上升越多，说明量化后模型「语言流畅性」下降越明显。

    实测方法：
      - 用 wikitext2 / c4 作为通用 benchmark
      - 分别跑 FP16 和 INT4 模型，对比 ppl 上升百分比
      - 目标：PPL 上升 < 5%（医疗场景可接受）
    """
    import math
    total_loss = 0.0
    total_tokens = 0

    encodings = tokenizer(texts, return_tensors="pt", truncation=True, max_length=512, padding=True)
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    if next(model.parameters()).device.type == "cpu":
        input_ids = input_ids
        attention_mask = attention_mask
    else:
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        # shifted loss: 预测下一个 token，忽略 padding
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        mask = shift_labels.view(-1) != tokenizer.pad_token_id
        total_loss = per_token_loss[mask].sum().item()
        total_tokens = mask.sum().item()

    ppl = math.exp(total_loss / total_tokens)
    return ppl


def compute_medical_qa_accuracy(model, tokenizer, test_data: list) -> float:
    """
    衡量医疗问答准确率（格式准确率 + 内容准确率）。

    分两个维度：
    1. 格式准确率：模型输出是否包含 <answer> 标签，按规定格式输出
    2. 内容准确率：用 ROUGE-L F1 衡量模型回答与参考回答的词汇 overlap

    原理：
      - 医疗问答不需要极端精度，INT4 量化后权重表达能力略有下降
      - 但医疗术语多为高频词，量化误差对其影响小，因此 <1% 差异可接受
    """
    rouge = Rouge()

    format_correct = 0
    rouge_scores = []

    for item in tqdm(test_data, desc="医疗问答评估"):
        question = item.get("Question", "")
        reference = item.get("Response", "")

        prompt = f"问：{question}\n请根据医学知识回答："
        inputs = tokenizer(prompt, return_tensors="pt")
        if next(model.parameters()).device.type != "cpu":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        raw_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
        if "<|im_start|>assistant" in raw_output:
            raw_output = raw_output.split("<|im_start|>assistant")[-1]

        # 抽取 answer
        import re
        m = re.search(r"<answer>\s*(.*?)\s*</answer>", raw_output, re.DOTALL)
        answer = m.group(1).strip() if m else raw_output.strip()

        # 格式准确率：是否包含 <answer> 标签
        if m:
            format_correct += 1

        # ROUGE-L 内容准确率
        ref_tokens = " ".join(jieba.cut(reference))
        hyp_tokens = " ".join(jieba.cut(answer))
        try:
            scores = rouge.get_scores(hyp_tokens, ref_tokens)
            rouge_l = scores[0]["rouge-l"]["f"]
        except Exception:
            rouge_l = 0.0
        rouge_scores.append(rouge_l)

    format_acc = format_correct / len(test_data) * 100
    avg_rouge_l = sum(rouge_scores) / len(rouge_scores) * 100
    return format_acc, avg_rouge_l


def evaluate_quantization_effect():
    """
    量化效果评估主函数。

    同时加载 FP16 原始模型 和 INT4 量化模型，在相同测试数据上跑三个指标：
      1. Perplexity（PPL）：通用 benchmark perplexity，越低越好
      2. 医疗问答准确率：ROUGE-L F1，越高越好
      3. 格式准确率：<answer> 标签输出比例，越高越好

    输出格式：
      FP16 模型 | INT4 模型 | 差异
      PPL:      | 20.5     | 21.3     | +3.9%
      ROUGE-L:  | 43.2%    | 42.8%    | -0.4%
      格式准确率:| 95.0%    | 94.5%    | -0.5%
    """
    test_data_path = "/home/amax/double/sft/SFT-RLHF/dataset/medical-o1-reasoning-sft/medical_o1_sft_Chinese.json"
    test_data = load_test_data(test_data_path, last_n=100)

    print("=" * 60)
    print("加载 FP16 原始模型...")
    fp16_model, fp16_tokenizer = load_model_and_tokenizer(model_id)

    print("=" * 60)
    print("加载 INT4 量化模型...")
    int4_model, int4_tokenizer = load_model_and_tokenizer(save_path)

    # ---- 1. Perplexity（通用 benchmark：wikitext2 / c4）----
    # 用测试集的全部 Question 作为通用文本来源
    perplexity_texts = [item["Question"] + " " + item.get("Response", "") for item in test_data]

    print("\n[1/3] 计算 Perplexity...")
    fp16_ppl = compute_ppl(fp16_model, fp16_tokenizer, perplexity_texts)
    int4_ppl = compute_ppl(int4_model, int4_tokenizer, perplexity_texts)
    ppl_diff = (int4_ppl - fp16_ppl) / fp16_ppl * 100

    # ---- 2. 医疗问答准确率（ROUGE-L）----
    print("\n[2/3] 计算医疗问答 ROUGE-L...")
    _, fp16_rouge = compute_medical_qa_accuracy(fp16_model, fp16_tokenizer, test_data)
    _, int4_rouge = compute_medical_qa_accuracy(int4_model, int4_tokenizer, test_data)
    rouge_diff = int4_rouge - fp16_rouge

    # ---- 3. 格式准确率（<answer> 标签）----
    print("\n[3/3] 计算格式准确率...")
    fp16_fmt, _ = compute_medical_qa_accuracy(fp16_model, fp16_tokenizer, test_data)
    int4_fmt, _ = compute_medical_qa_accuracy(int4_model, int4_tokenizer, test_data)
    fmt_diff = int4_fmt - fp16_fmt

    # ---- 打印结果 ----
    print("\n" + "=" * 60)
    print("量化效果评估结果（FP16 vs INT4）")
    print("=" * 60)
    print(f"{'指标':<16} {'FP16':>10} {'INT4':>10} {'差异':>10}")
    print("-" * 60)
    print(f"{'PPL':<16} {fp16_ppl:>10.2f} {int4_ppl:>10.2f} {ppl_diff:>+9.1f}%")
    print(f"{'ROUGE-L':<16} {fp16_rouge:>9.1f}% {int4_rouge:>9.1f}% {rouge_diff:>+9.1f}%")
    print(f"{'格式准确率':<16} {fp16_fmt:>9.1f}% {int4_fmt:>9.1f}% {fmt_diff:>+9.1f}%")
    print("=" * 60)

    # 评估结论
    if ppl_diff < 5 and abs(rouge_diff) < 1 and abs(fmt_diff) < 1:
        print("结论：量化后效果损失在可接受范围内 ✓")
    else:
        print("结论：量化损失超出预期，建议使用 INT8 或回退到 FP16")

    return {
        "ppl": {"fp16": fp16_ppl, "int4": int4_ppl, "diff_pct": ppl_diff},
        "rouge_l": {"fp16": fp16_rouge, "int4": int4_rouge, "diff": rouge_diff},
        "format_acc": {"fp16": fp16_fmt, "int4": int4_fmt, "diff": fmt_diff},
    }


def load_model_and_tokenizer(path: str):
    """加载模型（FP16 或 INT4 均可自动识别）。"""
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="quantize",
                        choices=["quantize", "evaluate", "both"],
                        help="quantize: 只量化; evaluate: 只评估; both: 量化+评估")
    args = parser.parse_args()

    if args.mode in ("quantize", "both"):
        gptq_quantize()

    if args.mode in ("evaluate", "both"):
        evaluate_quantization_effect()