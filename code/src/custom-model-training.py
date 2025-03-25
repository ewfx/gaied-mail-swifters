import pandas as pd
import torch
import os
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import joblib
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, 'dataset', "dataset.xlsx")

# Load dataset
data = pd.read_excel(dataset_path)

# Data Preprocessing
data.columns = data.columns.str.lower()
data.drop_duplicates(inplace=True)
data.fillna('Unknown', inplace=True)
data = data.astype(str)
data = data.applymap(lambda x: x.strip())

data['email_attachment'] = data['email_attachment'].str.replace(r'[^\w\s]', '', regex=True)

# Corrected Label Formatting
data['label'] = data['requesttype'] + "|||" + data['subrequesttype']

data = data[data['email_attachment'].str.strip() != ""]

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    data['email_attachment'], data['label'], test_size=0.2, random_state=42, stratify=data['label']
)

# Convert to Hugging Face dataset format
train_data = Dataset.from_pandas(pd.DataFrame({"text": X_train, "label": Y_train}))
test_data = Dataset.from_pandas(pd.DataFrame({"text": X_test, "label": Y_test}))

# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Tokenization function
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

train_data = train_data.map(tokenize, batched=True)
test_data = test_data.map(tokenize, batched=True)

# Corrected Label Encoding Using LabelEncoder
label_encoder = LabelEncoder()
Y_train_encoded = label_encoder.fit_transform(Y_train)
Y_test_encoded = label_encoder.transform(Y_test)

train_data = train_data.map(lambda x: {"label": label_encoder.transform([x["label"]])[0]})
test_data = test_data.map(lambda x: {"label": label_encoder.transform([x["label"]])[0]})

# Load DistilBERT model
num_labels = len(label_encoder.classes_)
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)
model.to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./bert_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir="./logs",
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save trained model, tokenizer, and label encoder
trained_model_path = os.path.join(current_dir, 'trained-models', 'bert_email_classifier')
model.save_pretrained(trained_model_path)
tokenizer.save_pretrained(trained_model_path)

labelmap_path = os.path.join(current_dir, 'trained-models', 'label_encoder.pkl')
joblib.dump(label_encoder, labelmap_path)

print("Model, tokenizer, and label encoder saved successfully in 'trained-models' directory.")
