# src/utils/dashboard_utils.py

import pandas as pd
import os
import json
from datetime import datetime

def save_feedback(feedback_dict, path="logs/feedback.csv"):
    """Append a feedback record to the feedback CSV."""
    df_new = pd.DataFrame([feedback_dict])
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(path, index=False)

def save_explanation(entry_dict, path="logs/explanations.csv"):
    """Append an explanation record."""
    df_new = pd.DataFrame([entry_dict])
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(path, index=False)

def log_query(query_record, path="logs/queries.csv"):
    """Append a query log with timestamp and optional response time."""
    df_new = pd.DataFrame([query_record])
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(path, index=False)

def save_attention_weights(weights_data, path="logs/attention_weights.json"):
    """Save graph attention weights as JSON."""
    with open(path, "w") as f:
        json.dump(weights_data, f, indent=2)

def log_metrics(metrics_record, path="logs/model_metrics.csv"):
    """Append model training metrics (loss, accuracy, etc.)."""
    df_new = pd.DataFrame([metrics_record])
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(path, index=False)

def timestamp_now():
    return datetime.utcnow().isoformat()
