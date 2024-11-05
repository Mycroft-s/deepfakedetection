import torch
from transformers import Trainer, TrainingArguments, default_data_collator
from Vision_Transformer import CustomImageDataset, load_model
from torch.utils.data import DataLoader

# Set the dataset path and model ID
img_dir = "F:/FF_Dataset/processed_frames"
model_id = "google/vit-base-patch16-224-in21k"

# Load the model and feature extractor
model, feature_extractor = load_model(model_id=model_id)

# Set device (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load training and testing datasets
train_dataset = CustomImageDataset(img_dir=img_dir + "/train", feature_extractor=feature_extractor)
test_dataset = CustomImageDataset(img_dir=img_dir + "/test", feature_extractor=feature_extractor)

# Create multi-threaded data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Define training parameters
training_args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=32,  # Increase batch size to make better use of GPU memory
    per_device_eval_batch_size=32,
    evaluation_strategy="epoch",
    num_train_epochs=1,
    save_strategy="epoch",
    logging_steps=50,
    learning_rate=5e-5,
    save_total_limit=2,  # Keep only the 2 most recent checkpoints
    remove_unused_columns=False,
    load_best_model_at_end=True,
    fp16=True  # Enable mixed precision training to speed up computation
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=default_data_collator
)

# Start training
trainer.train()

# Save the final model
trainer.save_model("./outputs/final_model")
print("Training complete. Model saved to ./outputs/final_model.")
