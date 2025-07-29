import numpy as np
import pandas as pd
import os
from tqdm import tqdm

import torch
import torch.nn as nn

import transformers
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          TrainingArguments,
                          Trainer,
                          AutoModelForMaskedLM,AutoConfig)

from datasets import load_dataset
from datasets import Dataset

from sklearn.metrics import (accuracy_score,
                             classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from transformers import DataCollatorWithPadding
from utils import *

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

def main(train_df, val_df, model_path, output_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id
    )

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    train_dataset = Dataset.from_pandas(train_df).map(tokenize_function, batched=True).remove_columns(['text'])
    val_dataset = Dataset.from_pandas(val_df).map(tokenize_function, batched=True).remove_columns(['text'])

    #define training arguments
    train_batch, val_batch = (256 ,256)
    lr = 1e-5
    betas = (0.9, 0.99)
    n_epochs = 20
    eps = 1e-6

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        per_device_train_batch_size=train_batch,
        per_device_eval_batch_size=val_batch,
        num_train_epochs=n_epochs,
        lr_scheduler_type="linear",
        optim="adamw_torch_fused",
        adam_beta1=betas[0],
        adam_beta2=betas[1],
        adam_epsilon=eps,
        logging_strategy="steps",
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model="f1",
        greater_is_better=True,
        bf16=True,
        push_to_hub=False,
        report_to="wandb",
        run_name="ALBERT-Base-Best",
    )

    #Create a Trainer instance
    trainer = Trainer(
        model=model,                         # The pre-trained model
        args=training_args,                  # Training arguments
        train_dataset=train_dataset,            # Tokenized training dataset
        eval_dataset=val_dataset, 
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()



if __name__ == "__main__":
    train_path = '/root/Prodigal-Take-Home/sentiment-take-home/sentiment-take-home/train_df.csv'
    val_path = '/root/Prodigal-Take-Home/sentiment-take-home/sentiment-take-home/val_df.csv'
    test_path = '/root/Prodigal-Take-Home/sentiment-take-home/sentiment-take-home/test_df.csv'

    conversations_path = '/root/Prodigal-Take-Home/sentiment-take-home/sentiment-take-home/conversations'

    train_df = create_dataset(train_path, conversations_path)
    val_df = create_dataset(val_path, conversations_path)
    test_df = create_dataset(test_path, conversations_path)

    main(train_df, val_df, model_path="albert/albert-base-v2", output_dir="albert-best")

