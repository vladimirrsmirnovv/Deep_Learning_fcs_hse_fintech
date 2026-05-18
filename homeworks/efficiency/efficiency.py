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
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="sdpa",
    )

    base_model.to(torch.bfloat16) # что тут не так?

    # PEFT configuration for LoRA
    # TODO: your code here

    # Wrap base model with LoRA
    # TODO: your code here

    # Any other optimizations?
    # TODO: your code here

    return model

def prepare_dataloader(dataset, tokenizer):

    # 1. Tokenize dataset
    # 2. Add truncation for 1024 due to limited GPT2 context window
    # 3. Create dataloader with collate_fn of DataCollatorForLanguageModeling

    # TODO: your code here

    batch_size = 1

    train_dataloader = DataLoader(tokenized_dataset, collate_fn=data_collator, batch_size=batch_size, shuffle=True)

    return train_dataloader

def train_model(model, train_dataloader, optimizer, num_epochs, device):

    # TODO: Что делает эта строка?
    # torch.set_float32_matmul_precision('high')

    total_loss = 0

    for epoch in range(num_epochs):
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for i, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )

            loss = outputs.loss

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Update progress bar
            if i % 10 == 0:
                # вызов .item() приводит к синхронизации CPU и GPU
                # это медленная операция, поэтому будем делать ее раз в 10 итераций
                loss_item = loss.item()
                total_loss = total_loss * 0.8 + loss_item * 0.2
                progress_bar.set_postfix({'loss': loss_item})

        print(f"Epoch {epoch + 1} moving average loss: {total_loss:.4f}")

    return total_loss


if __name__ == "__main__":
    #
    # ❗️ Содержимое этого блока менять не нужно! ❗️
    # ❗️ Изменения, которые вы сделаете не будут учтены в автогрейдере
    # ❗️ В автогрейдере будут использоваться только функции, определенные выше
    #

    os.environ['WANDB_DISABLED'] = 'true'

    # Load tokenizer and model
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset('HuggingFaceFW/fineweb', data_files=['data/CC-MAIN-2024-10/000_00000.parquet'])
    dataset['train'] = dataset['train'].select(range(6000))

    train_dataloader = prepare_dataloader(dataset, tokenizer)

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = prepare_model(model_name)

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            for param_name, param in module.named_parameters(recurse=False):
                assert param.dtype == torch.float32, (
                    f"LayerNorm param {name}.{param_name} expected float32, got {param.dtype}"
                )
        else:
            for param_name, param in module.named_parameters(recurse=False):
                assert param.dtype == torch.bfloat16, (
                    f"Param {name}.{param_name} expected bfloat16, got {param.dtype}"
                )

    model = model.to(device)
    model.train()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Training loop
    num_epochs = 1
    moving_avg_loss = train_model(model, train_dataloader, optimizer, num_epochs, device=device)
    print(f"Moving average loss: {moving_avg_loss:.4f}")
    assert moving_avg_loss < 2.0
