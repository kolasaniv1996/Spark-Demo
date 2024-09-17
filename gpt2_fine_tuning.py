import os
import time
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from pyspark.sql import SparkSession

# Initialize SparkSession
spark = SparkSession.builder.appName("GPT2FineTuning").getOrCreate()

# Check if GPUs are available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directory to save checkpoints
checkpoint_dir = '/home/local/data/checkpoint'
os.makedirs(checkpoint_dir, exist_ok=True)

# Load the distilgpt2 model and tokenizer from Hugging Face
model_name = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Option 1: Use eos_token as pad_token
tokenizer.pad_token = tokenizer.eos_token

# Load the smaller model (distilgpt2)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.to(device)

# Load a small dataset for fine-tuning (e.g., wikitext dataset)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask"])

# Create DataLoader for fine-tuning
train_dataloader = DataLoader(tokenized_dataset, batch_size=4, shuffle=True)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
def fine_tune_model(epoch, model, dataloader, optimizer, device, checkpoint_dir):
    model.train()
    total_loss = 0

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        if step % 10 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

    # Save the model checkpoint after each epoch
    checkpoint_path = os.path.join(checkpoint_dir, f"gpt2_checkpoint_epoch_{epoch}.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

# Run the fine-tuning process
epochs = 3
for epoch in range(epochs):
    fine_tune_model(epoch, model, train_dataloader, optimizer, device, checkpoint_dir)
    time.sleep(5)  # Simulate some delay between epochs

# Stop SparkSession
spark.stop()
