import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, accuracy_score


def log_experiment(predictions, ground_truth, features:list=None, additional:dict=None, model_path_val=None, model_path_full=None, results_dir="results", img_dir="results/img"):
    """
    Logs various metrics (F1 score, precision, recall, accuracy) and confusion matrix
    to a dynamically named markdown file and saves the confusion matrix plot as an image.

    Args:
        predictions (array-like): Predicted labels.
        ground_truth (array-like): True labels.
        results_dir (str): Directory to save the markdown file.
        img_dir (str): Directory to save the confusion matrix image.
    """
    # Ensure directories exist
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    # Calculate metrics
    f1 = f1_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions)
    recall = recall_score(ground_truth, predictions)
    accuracy = accuracy_score(ground_truth, predictions)
    cm = confusion_matrix(ground_truth, predictions)

    # Determine next file index
    existing_files = sorted([f for f in os.listdir(results_dir) if f.startswith("exp_") and f.endswith(".md")])
    next_index = (int(existing_files[-1].split("_")[1].split(".")[0]) + 1) if existing_files else 1
    file_index = f"{next_index:02d}"

    # File paths
    md_file_path = os.path.join(results_dir, f"exp_{file_index}.md")
    img_file_path = os.path.join(img_dir, f"confusion_matrix_{file_index}.png")

    # Save confusion matrix plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(img_file_path)
    plt.close()

    # Write to markdown file
    with open(md_file_path, "w") as md_file:
        md_file.write(f"# Experiment {file_index}\n\n")
        md_file.write(f"## Metrics\n")
        md_file.write(f"- F1 Score: {f1:.3f}\n")
        md_file.write(f"- Precision: {precision:.3f}\n")
        md_file.write(f"- Recall: {recall:.3f}\n")
        md_file.write(f"- Accuracy: {accuracy:.3f}\n\n")

        md_file.write(f"## Model Path\n")
        if model_path_val:
            md_file.write(f"- Model Path: {model_path_val}\n\n")
        else:
            md_file.write(f"- Model Path: Not provided\n\n")
        if model_path_full:
            md_file.write(f"- Full Model Path: {model_path_full}\n\n")
        else:
            md_file.write(f"- Full Model Path: Not provided\n\n")
        md_file.write(f"## Confusion Matrix\n")
        md_file.write(f"![Confusion Matrix](img/confusion_matrix_{file_index}.png)\n")
        md_file.write(f"\n## Additional Information\n")
        if additional:
            for key, value in additional.items():
                md_file.write(f"- {key}: {value}\n")
        else:
            md_file.write(f"- No additional information provided.\n")

                # Features
        md_file.write(f"## Features\n")
        for idx, feature in enumerate(features):
            md_file.write(f"- {idx} Feature: {feature}\n")

    print(f"Experiment logged: {md_file_path}")

