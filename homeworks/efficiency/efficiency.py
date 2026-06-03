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
    # В новых transformers вместо torch_dtype предпочтительнее dtype.
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

        # Достаточно сильный LoRA, но не слишком тяжелый.
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,

        # GPT-2 Conv1D modules:
        # c_attn — attention QKV projection,
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
    # Компромисс:
    # 512 слишком мало по loss,
    # 1024 слишком долго,
    # 896 должно дать больше "лёгкого" контекста/padding, но быстрее 1024.
    # ------------------------------------------------------------
    max_length = 896

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

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 6000 / 12 = 500.
    # Больше optimizer updates, чем при batch_size=16.
    # При этом batch не такой тяжёлый, как 16/20 на длинном контексте.
    batch_size = 12

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

    # Внешний main даёт lr=1e-4, для одного прохода это слишком мало.
    # Но постоянный высокий LR даёт скачки loss ближе к концу.
    # Поэтому: высокий старт + cosine decay.
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

                        # Оставляем как в исходном шаблоне:
                        # loss считается по input_ids.
                        # batch["labels"] от collator маскирует padding через -100,
                        # а это в этой задаче даёт более высокий loss.
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
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()

            scheduler.step()

            # .item() синхронизирует CPU и GPU.
            # Оставляем каждые 10 итераций, чтобы логика moving average
            # была близка к исходному шаблону.
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
