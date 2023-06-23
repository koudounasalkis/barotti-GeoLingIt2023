"""
python run_inference_tasksab.py \
    --ds_train_path ./data/train_ab.csv \
    --ds_valid_path ./data/dev_ab.csv \
    --output_folder results_tasksab \
    --batch 8 \
    --num_epochs 10 \
    --lr 5e-5 \
    --model dbmdz/bert-base-italian-uncased \
"""



import os

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, AutoModel
from transformers import TrainingArguments, DefaultDataCollator, Trainer
from sklearn.preprocessing import LabelEncoder
from utils import compute_metrics, DualDataset, DualTrainer
from argparse import ArgumentParser

from dual_model import DualModel

import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":

    parser = ArgumentParser(description="EvalITA - GeoLingit")
    parser.add_argument(
        "--ds_train_path",
        help="Train dataset",
        default="./data/train_ab.csv",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--ds_valid_path",
        help="Valid dataset",
        default="./data/dev_ab.csv",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--output_folder",
        help="Output folder",
        default="results_tasksab/",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--batch",
        help="Batch size",
        default=8,
        required=False,
        type=int,
    )
    parser.add_argument(
        "--num_epochs",
        help="Number of training epochs",
        default=20,
        required=False,
        type=int,
    )
    parser.add_argument(
        "--lr",
        help="Learning rate",
        default=5e-5,
        required=False,
        type=float,
    )
    parser.add_argument(
        "--weight_decay",
        help="Weight decay",
        default=0.01,
        required=False,
        type=float,
    )
    parser.add_argument(
        "--warmup_ratio",
        help="Warmup ratio",
        default=0.06,
        required=False,
        type=float,
    )
    parser.add_argument(
        "--model",
        help="Model to use",
        default="dbmdz/bert-base-italian-uncased",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--ckpt",
        help="Model checkpoint to evaluate",
        default="best-ckpt",
        required=False,
        type=str,
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Parameters
    ds_train_path = args.ds_train_path  # e.g., './data/train_ab.csv'
    ds_valid_path = args.ds_valid_path  # e.g., './data/dev_ab.csv'
    output_folder = args.output_folder  # e.g., './inference/'

    BSZ = args.batch                    # e.g., 8
    EPOCHS = args.num_epochs            # e.g., 5
    LR = args.lr                        # e.g., 5e-5 
    WEIGHT_DECAY = args.weight_decay    # e.g., 0.01
    WARMUP_RATIO = args.warmup_ratio    # e.g., 0.06
    model_path = args.model             # e.g., dbmdz/bert-base-italian-uncased
    model_ckpt = args.ckpt              # e.g., best-ckpt/pytorch_model.bin
    
    ## Output folder
    os.makedirs(output_folder, exist_ok=True)

    labels = pd.read_csv(ds_train_path)['region']
    le = LabelEncoder()
    le.fit(labels)
    num_labels = len(list(le.classes_))

    ## Model
    model = DualModel(model_path, num_classes=num_labels, validation=False)
    model.load_state_dict(torch.load(model_ckpt))
    model.eval()

    ## Train dataset
    train_ds = DualDataset(
        ds_train_path, 
        model_path, 
        le, 
        split="train")

    ## Dev dataset
    dev_ds = DualDataset(
        ds_valid_path, 
        model_path, 
        le, 
        split="dev")

    ## Training arguments
    training_args = TrainingArguments(
        output_dir=output_folder,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        per_device_train_batch_size=BSZ,
        per_device_eval_batch_size=BSZ,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        gradient_accumulation_steps=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        metric_for_best_model="dist_km",
        save_total_limit=5,
        remove_unused_columns=False)

    ## Data collator
    data_collator = DefaultDataCollator()

    ## DualTrainer
    trainer = DualTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=compute_metrics,
        data_collator=data_collator)

    ## Inference
    trainer.evaluate()
    predictions = trainer.predict(dev_ds)
    with open(f"{output_folder}/test_preds_taskb_multihead_besta.csv", "w") as f:
        f.write("id,latitude,longitude\n")
        for i, pred in enumerate(predictions.predictions[1]):
            f.write(f"{i+13670},{pred[0]},{pred[1]}\n")