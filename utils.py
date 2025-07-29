import numpy as np
import pandas as pd
import os
from sklearn.metrics import (accuracy_score,
                             f1_score)

# Define the label mapping
id2label_borrower = {0: "negative", 1: "positive"}
label2id_borrower = {"negative": 0, "positive": 1}

id2label_agent = {0: 'Promise - Payment plan', 1: 'Promise - Settlement in full',\
                   2: 'No Pay - Cant pay now', 3: 'No Pay - Not right now',\
                      4: 'No Pay - Dispute', 5: 'Promise - Payment in full',\
                          6: 'No Pay - Bankruptcy', 7: 'Promise - One time payment',\
                              8: 'Promise - Settlement in payments', 9: 'Payment Plan Modification',\
                                  10: 'No Pay - Cancel payment plan'}
label2id_agent = {v: k for k, v in id2label_agent.items()}


def create_dataset(file_path, conversations_path, label2id, col):
    df = pd.read_csv(file_path)

    def add_text_column(df, conversations_path):
        """
        Adds a 'text' column to the DataFrame by reading text from files.

        Args:
            df (pd.DataFrame): The DataFrame to modify.
            conversations_path (str): The path to the directory containing the text files.

        Returns:
            pd.DataFrame: The DataFrame with the added 'text' column.
        """
        df['text'] = df['_id'].apply(lambda x: open(os.path.join(conversations_path, f"{x}.txt")).read())
        return df

    df = add_text_column(df, conversations_path)
    df['labels'] = df[col].map(label2id)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = df[['text', 'labels']]

    return df

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # The predictions are raw logits
    predictions = np.argmax(predictions, axis=1)

    # Calculate weighted F1 score
    f1 = f1_score(labels, predictions, average="weighted")

    # Calculate accuracy
    accuracy = accuracy_score(labels, predictions)

    # Return both metrics in a dictionary
    return {"f1": f1, "accuracy": accuracy}