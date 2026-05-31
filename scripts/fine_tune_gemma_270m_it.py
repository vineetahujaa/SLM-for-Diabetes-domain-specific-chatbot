#!/usr/bin/env python3
"""Fine-tune Gemma 270M IT for the diabetes-care chatbot with LoRA SFT.

This script is the cleaned, reusable version of the original notebook workflow:
1. Load a CSV dataset with prompt/target columns.
2. Shuffle and split it into train/validation sets.
3. Fine-tune google/gemma-3-270m-it with LoRA using TRL SFTTrainer.
4. Save the LoRA adapter and tokenizer.
5. Optionally merge the adapter into the base Hugging Face model.

The app itself uses a GGUF file through llama.cpp. After merging, convert the merged
Hugging Face model to GGUF and quantize it to Q8_0 with llama.cpp.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning for google/gemma-3-270m-it using a CSV dataset."
    )
    parser.add_argument("--model-id", default="google/gemma-3-270m-it", help="Base HF model ID.")
    parser.add_argument("--train-csv", default="train_mixed.csv", help="Input CSV dataset path.")
    parser.add_argument("--prepared-csv", default="qa_cleaned_shuffled.csv", help="Shuffled CSV output path.")
    parser.add_argument("--prompt-col", default="prompt", help="CSV column containing user prompts/questions.")
    parser.add_argument("--target-col", default="target", help="CSV column containing assistant answers.")
    parser.add_argument("--adapter-dir", default="gemma-diabetes-final", help="Output LoRA adapter directory.")
    parser.add_argument("--checkpoint-dir", default="gemma-diabetes", help="Trainer checkpoint directory.")
    parser.add_argument("--merged-dir", default="gemma-diabetes-merged-hf", help="Merged HF model output directory.")
    parser.add_argument("--test-size", type=float, default=0.10, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length.")
    parser.add_argument("--epochs", type=float, default=3, help="Training epochs.")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-device train batch size.")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--eval-steps", type=int, default=500, help="Evaluation interval in steps.")
    parser.add_argument("--save-steps", type=int, default=500, help="Checkpoint save interval in steps.")
    parser.add_argument("--save-total-limit", type=int, default=3, help="Maximum checkpoints to keep.")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN", ""), help="Hugging Face token, or set HF_TOKEN.")
    parser.add_argument("--no-shuffle", action="store_true", help="Do not shuffle the input CSV before training.")
    parser.add_argument("--merge", action="store_true", help="Merge the trained adapter into the base HF model.")
    parser.add_argument(
        "--merge-device-map",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device map used while merging LoRA into the base model.",
    )
    return parser.parse_args()


def require_training_dependencies() -> dict[str, Any]:
    try:
        import pandas as pd
        import torch
        from datasets import Dataset
        from huggingface_hub import login
        from peft import LoraConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit(
            "Missing fine-tuning dependency. Install them with:\n"
            "  pip install -r requirements-finetune.txt"
        ) from exc

    return {
        "pd": pd,
        "torch": torch,
        "Dataset": Dataset,
        "login": login,
        "LoraConfig": LoraConfig,
        "PeftModel": PeftModel,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "SFTConfig": SFTConfig,
        "SFTTrainer": SFTTrainer,
    }


def prepare_dataset(args: argparse.Namespace, deps: dict[str, Any]) -> Any:
    pd = deps["pd"]
    Dataset = deps["Dataset"]

    csv_path = Path(args.train_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [col for col in (args.prompt_col, args.target_col) if col not in df.columns]
    if missing:
        raise ValueError(f"Missing CSV column(s): {', '.join(missing)}. Found: {list(df.columns)}")

    df = df[[args.prompt_col, args.target_col]].dropna()
    df[args.prompt_col] = df[args.prompt_col].astype(str).str.strip()
    df[args.target_col] = df[args.target_col].astype(str).str.strip()
    df = df[(df[args.prompt_col] != "") & (df[args.target_col] != "")]

    if df.empty:
        raise ValueError("No usable training rows after cleaning empty prompt/target values.")

    if not args.no_shuffle:
        df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        df.to_csv(args.prepared_csv, index=False)
        print(f"Saved shuffled dataset to: {args.prepared_csv}")

    def format_row(row: Any) -> dict[str, list[dict[str, str]]]:
        return {
            "prompt": [{"role": "user", "content": str(row[args.prompt_col]).strip()}],
            "completion": [{"role": "assistant", "content": str(row[args.target_col]).strip()}],
        }

    formatted = [format_row(row) for _, row in df.iterrows()]
    dataset = Dataset.from_list(formatted).train_test_split(test_size=args.test_size, seed=args.seed)

    print(dataset["train"][0])
    print(f"Train: {len(dataset['train'])} | Val: {len(dataset['test'])}")
    return dataset


def best_dtype(torch: Any) -> Any:
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def train(args: argparse.Namespace, deps: dict[str, Any]) -> None:
    torch = deps["torch"]
    login = deps["login"]
    LoraConfig = deps["LoraConfig"]
    AutoModelForCausalLM = deps["AutoModelForCausalLM"]
    AutoTokenizer = deps["AutoTokenizer"]
    SFTConfig = deps["SFTConfig"]
    SFTTrainer = deps["SFTTrainer"]

    if args.hf_token:
        login(args.hf_token)

    dataset = prepare_dataset(args, deps)
    dtype = best_dtype(torch)
    use_bf16 = dtype == torch.bfloat16
    use_fp16 = dtype == torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    print("Base model loaded.")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    config = SFTConfig(
        output_dir=args.checkpoint_dir,
        max_length=args.max_length,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=use_bf16,
        fp16=use_fp16,
        completion_only_loss=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        peft_config=lora_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )
    trainer.train()

    trainer.model.save_pretrained(args.adapter_dir)
    tokenizer.save_pretrained(args.adapter_dir)
    print(f"Saved LoRA adapter to: {args.adapter_dir}")


def merge_adapter(args: argparse.Namespace, deps: dict[str, Any]) -> None:
    torch = deps["torch"]
    PeftModel = deps["PeftModel"]
    AutoModelForCausalLM = deps["AutoModelForCausalLM"]
    AutoTokenizer = deps["AutoTokenizer"]

    adapter_dir = Path(args.adapter_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    if args.merge_device_map == "cpu":
        dtype = torch.float32
    else:
        dtype = best_dtype(torch)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map=args.merge_device_map,
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = model.merge_and_unload()
    merged.save_pretrained(args.merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.merged_dir)
    print(f"Merged HF model saved to: {args.merged_dir}")


def main() -> None:
    args = parse_args()
    deps = require_training_dependencies()
    train(args, deps)
    if args.merge:
        merge_adapter(args, deps)


if __name__ == "__main__":
    main()
