import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score

# 1. Load and Filter Dataset
# GoEmotions has 28 labels; we map them to your specific requirements
dataset = load_dataset("go_emotions", "simplified")

# Define target emotions and their IDs in GoEmotions
target_map = {0: "admiration", 1: "amusement", 2: "anger", 3: "annoyance", 4: "approval", 
              5: "caring", 6: "confusion", 7: "curiosity", 8: "desire", 9: "disappointment", 
              10: "disapproval", 11: "disgust", 12: "embarrassment", 13: "excitement", 
              14: "fear", 15: "gratitude", 16: "grief", 17: "joy", 18: "love", 19: "nervousness", 
              20: "optimism", 21: "outrage", 22: "realization", 23: "relief", 24: "remorse", 
              25: "sadness", 26: "surprise", 27: "neutral"}

# 2. Preprocessing & Tokenization
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

# We'll use a subset for speed in this beginner example
train_ds = dataset["train"].shuffle(seed=42).select(range(2000))
test_ds = dataset["test"].shuffle(seed=42).select(range(500))

train_tokenized = train_ds.map(tokenize, batched=True)
test_tokenized = test_ds.map(tokenize, batched=True)

# 3. Model Initialization
# We've set num_labels to 28 to match the simplified GoEmotions labels
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=28)

# 4. Evaluation Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

# 5. Training
training_args = TrainingArguments(
    output_dir="./emotion_model",
    num_train_epochs=2,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    weight_decay=0.01,
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    tokenizer=tokenizer
)

if __name__ == "__main__":
    trainer.train()
    # Save the model for the UI
    model.save_pretrained("./final_model")
    tokenizer.save_pretrained("./final_model")
    print("Training complete! Model saved to ./final_model")