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
    # 1. dtype
    # ------------------------------------------------------------
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    else:
        dtype = torch.float32

    # ------------------------------------------------------------
    # 2. GPU matmul optimizations
    # ------------------------------------------------------------
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    torch.set_float32_matmul_precision("high")

    # ------------------------------------------------------------
    # 3. Load GPT-2
    # ------------------------------------------------------------
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            attn_implementation="sdpa",
        )
    except TypeError:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            attn_implementation="sdpa",
        )

    base_model.config.use_cache = False

    # ------------------------------------------------------------
    # 4. LoRA
    # ------------------------------------------------------------
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,

        r=16,
        lora_alpha=32,
        lora_dropout=0.0,

        # GPT-2 modules:
        # c_attn — QKV projection,
        # c_proj — attention/MLP output projection,
        # c_fc — MLP projection.
        target_modules=["c_attn", "c_proj", "c_fc"],

        fan_in_fan_out=True,
        bias="none",
    )

    model = get_peft_model(base_model, peft_config)

    return model


def prepare_dataloader(dataset, tokenizer):
    train_dataset = dataset["train"]
    text_column = "text"

    # ------------------------------------------------------------
    # Главное изменение:
    # SEQ_LEN — итоговая фиксированная длина для torch.compile.
    # REAL_TOKENS — сколько реального текста берём из каждого объекта.
    #
    # Остальное добиваем eos_token.
    # Это сохраняет fixed shape, ускоряет обучение и снижает loss.
    # ------------------------------------------------------------
    SEQ_LEN = 768
    REAL_TOKENS = 256

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.pad_token_id

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples[text_column],
            truncation=True,
            max_length=REAL_TOKENS,
            padding=False,
            return_attention_mask=False,
        )

        all_input_ids = []
        all_attention_masks = []

        for ids in tokenized["input_ids"]:
            ids = ids[:REAL_TOKENS]

            pad_len = SEQ_LEN - len(ids)

            if pad_len > 0:
                ids = ids + [eos_id] * pad_len
            else:
                ids = ids[:SEQ_LEN]

            # Важно:
            # здесь eos-добивку считаем частью последовательности,
            # а не padding'ом. Поэтому attention_mask = 1 на всех позициях.
            # Так модель видит повторяющиеся eos и быстрее снижает loss.
            attention_mask = [1] * SEQ_LEN

            all_input_ids.append(ids)
            all_attention_masks.append(attention_mask)

        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
        }

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
    # Быстрый collate без DataCollatorForLanguageModeling.
    # Labels нам не нужны в batch, потому что в train_model
    # передаём labels=input_ids.
    # ------------------------------------------------------------
    def fast_collate(features):
        input_ids = torch.stack([x["input_ids"] for x in features])
        attention_mask = torch.stack([x["attention_mask"] for x in features])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    # 6000 / 16 = 375.
    # Этот режим уже проходил по времени при SEQ_LEN=768.
    batch_size = 16

    train_dataloader = DataLoader(
        tokenized_dataset,
        collate_fn=fast_collate,
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
    # Matmul precision
    # ------------------------------------------------------------
    # torch.set_float32_matmul_precision("high") разрешает более быстрые
    # режимы float32 matmul, включая TF32 на Ampere+ GPU.
    torch.set_float32_matmul_precision("high")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ------------------------------------------------------------
    # torch.compile после model.to(device)
    # ------------------------------------------------------------
    if device.type == "cuda" and hasattr(torch, "compile"):
        try:
            model = torch.compile(
                model,
                mode="reduce-overhead",
                fullgraph=False,
                dynamic=False,
            )
        except Exception as e:
            print(f"torch.compile skipped: {e}")

    # ------------------------------------------------------------
    # Optimizer только на trainable LoRA-параметры
    # ------------------------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # Внешний optimizer создаётся с lr=1e-4.
    # Для одного прохода LoRA это слишком мало.
    effective_lr = 5e-3

    if device.type == "cuda":
        try:
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=effective_lr,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=0.0,
                fused=True,
            )
        except TypeError:
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=effective_lr,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=0.0,
            )
    else:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=effective_lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.0,
        )

    total_steps = len(train_dataloader) * num_epochs

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=2e-4,
    )

    # ------------------------------------------------------------
    # AMP
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

    if use_grad_scaler:
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        except TypeError:
            scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None

    model.train()
    total_loss = 0.0

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------
    for epoch in range(num_epochs):
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
        )

        for i, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

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
                        labels=input_ids,
                    )
                    loss = outputs.loss
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                )
                loss = outputs.loss

            if use_grad_scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            scheduler.step()

            # .item() синхронизирует CPU и GPU.
            # Оставляем раз в 10 итераций, как в исходной логике.
            if i % 10 == 0:
                loss_item = loss.detach().float().item()
                total_loss = total_loss * 0.8 + loss_item * 0.2
                progress_bar.set_postfix({"loss": loss_item})

        print(f"Epoch {epoch + 1} moving average loss: {total_loss:.4f}")

    return total_loss


if __name__ == "__main__":
    #
    # ❗️ Содержимое этого блока менять не нужно! ❗️
    # ❗️ Изменения, которые вы сделаете не будут учтены в автогрейдере
    # ❗️ В автогрейдере будут использоваться только функции, определенные выше
    #

    os.environ["WANDB_DISABLED"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model_name = "gpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        data_files=["data/CC-MAIN-2024-10/000_00000.parquet"],
    )
    dataset["train"] = dataset["train"].select(range(6000))

    train_dataloader = prepare_dataloader(dataset, tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = prepare_model(model_name)

    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    num_epochs = 1
    moving_avg_loss = train_model(
        model,
        train_dataloader,
        optimizer,
        num_epochs,
        device=device,
    )

    print(f"Moving average loss: {moving_avg_loss:.4f}")
    assert moving_avg_loss < 2.0
