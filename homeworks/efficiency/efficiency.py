from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import torch
from torch.utils.data import DataLoader
from peft import get_peft_model, LoraConfig, TaskType
from transformers import DataCollatorForLanguageModeling
import os
from tqdm import tqdm


def prepare_model(model_name):
    # ------------------------------------------------------------
    # 1. Выбираем dtype
    # ------------------------------------------------------------
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    else:
        dtype = torch.float32

    # ------------------------------------------------------------
    # 2. Общие ускорения для matmul
    # ------------------------------------------------------------
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    torch.set_float32_matmul_precision("high")

    # ------------------------------------------------------------
    # 3. Загружаем GPT-2
    # ------------------------------------------------------------
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )

    # Для обучения use_cache не нужен.
    # Более того, он часто конфликтует с оптимизациями обучения.
    base_model.config.use_cache = False

    # ------------------------------------------------------------
    # 4. LoRA config
    # ------------------------------------------------------------
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,

        # Небольшой rank: быстрее и достаточно для задачи.
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,

        # Для GPT-2 основные проекции называются c_attn и c_proj.
        # c_attn отвечает за QKV-проекцию.
        # c_proj встречается и в attention, и в MLP-блоке.
        target_modules=["c_attn", "c_proj"],

        # GPT-2 использует Conv1D из transformers,
        # для него нужен fan_in_fan_out=True.
        fan_in_fan_out=True,

        bias="none",
    )

    model = get_peft_model(base_model, peft_config)

    # ------------------------------------------------------------
    # 5. torch.compile
    # ------------------------------------------------------------
    # Важно: compile выигрывает только если shapes стабильные.
    # Поэтому в dataloader ниже будет padding="max_length".
    if torch.cuda.is_available() and hasattr(torch, "compile"):
        model = torch.compile(
            model,
            mode="reduce-overhead",
            fullgraph=False,
            dynamic=False,
        )

    return model


def prepare_dataloader(dataset, tokenizer):
    # ------------------------------------------------------------
    # 1. Берём train split
    # ------------------------------------------------------------
    train_dataset = dataset["train"]

    # FineWeb обычно содержит колонку "text".
    text_column = "text"

    # ------------------------------------------------------------
    # 2. Фиксированный max_length
    # ------------------------------------------------------------
    # GPT-2 context window = 1024.
    # Важно использовать padding="max_length", иначе batch shapes будут разные,
    # и torch.compile будет тратить время на recompilation.
    max_length = 512


    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_attention_mask=True,
        )

    tokenized_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing",
    )

    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask"],
    )

    # ------------------------------------------------------------
    # 3. Collator для causal language modeling
    # ------------------------------------------------------------
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 6000 / 16 = 375, то есть последнего неполного батча не будет.
    # Это важно для torch.compile: batch dimension тоже остаётся стабильной.
    batch_size = 24

    train_dataloader = DataLoader(
        tokenized_dataset,
        collate_fn=data_collator,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )

    return train_dataloader


def train_model(model, train_dataloader, optimizer, num_epochs, device):
    # ------------------------------------------------------------
    # Что делает эта строка?
    # ------------------------------------------------------------
    # torch.set_float32_matmul_precision("high") разрешает PyTorch
    # использовать более быстрые режимы матричного умножения для float32,
    # в частности TF32 на NVIDIA Ampere+.
    # Это ускоряет matmul, иногда с небольшой потерей численной точности.
    torch.set_float32_matmul_precision("high")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ------------------------------------------------------------
    # 1. Пересобираем optimizer только на trainable parameters
    # ------------------------------------------------------------
    # Снаружи optimizer создаётся как AdamW(model.parameters()).
    # Но у LoRA большая часть параметров frozen, поэтому быстрее оставить
    # в optimizer только параметры с requires_grad=True.
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if len(trainable_params) > 0:
        old_group = optimizer.param_groups[0]

        lr = old_group.get("lr", 1e-4)
        betas = old_group.get("betas", (0.9, 0.999))
        eps = old_group.get("eps", 1e-8)
        weight_decay = old_group.get("weight_decay", 0.01)

        if device.type == "cuda":
            try:
                optimizer = torch.optim.AdamW(
                    trainable_params,
                    lr=lr,
                    betas=betas,
                    eps=eps,
                    weight_decay=weight_decay,
                    fused=True,
                )
            except TypeError:
                optimizer = torch.optim.AdamW(
                    trainable_params,
                    lr=lr,
                    betas=betas,
                    eps=eps,
                    weight_decay=weight_decay,
                )
        else:
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
            )

    # ------------------------------------------------------------
    # 2. AMP dtype
    # ------------------------------------------------------------
    use_amp = device.type == "cuda"

    if use_amp and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        use_grad_scaler = False
    elif use_amp:
        amp_dtype = torch.float16
        use_grad_scaler = True
    else:
        amp_dtype = torch.float32
        use_grad_scaler = False

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_grad_scaler,
    )

    model.train()
    total_loss = 0.0

    # ------------------------------------------------------------
    # 3. Training loop
    # ------------------------------------------------------------
    for epoch in range(num_epochs):
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
        )

        for i, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            # DataCollatorForLanguageModeling создаёт labels.
            # В labels padding-токены заменяются на -100,
            # поэтому они не входят в loss.
            labels = batch.get("labels", None)
            if labels is not None:
                labels = labels.to(device, non_blocking=True)
            else:
                labels = input_ids

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(
                    device_type="cuda",
                    dtype=amp_dtype,
                    enabled=True,
                ):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

            if use_grad_scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # .item() синхронизирует CPU и GPU, поэтому редко.
            if i % 10 == 0:
                loss_item = loss.detach().float().item()
                total_loss = total_loss * 0.8 + loss_item * 0.2
                progress_bar.set_postfix({"loss": loss_item})

        print(f"Epoch {epoch + 1} moving average loss: {total_loss:.4f}")

    return total_loss
