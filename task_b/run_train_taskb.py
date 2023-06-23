"""
python run_train_taskb.py \
    --ds_train_path ./train_b.tsv \
    --ds_valid_path ./dev_b.tsv \
    --output_folder results_taskb/ \
    --batch 8 \
    --num_epochs 6 \
    --lr 5e-5 \
    --model dbmdz/bert-base-italian-uncased 
"""

import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers import TrainingArguments, DefaultDataCollator, Trainer
from sklearn.preprocessing import LabelEncoder
from utils import compute_metrics, RegionDataset, dist_km
from argparse import ArgumentParser


from torch import nn
from transformers import Trainer

"""
    RegressionTrainer is a subclass of Trainer that allows to compute the loss function for regression tasks.
    It is used in the training phase.
"""
class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss_fct = torch.nn.MSELoss()
        loss = loss_fct(logits, labels).mean()
        return (loss, outputs) if return_outputs else loss


if __name__ == "__main__":

    parser = ArgumentParser(description="EvalITA - GeoLingit")
    parser.add_argument(
        "--ds_train_path",
        help="Train dataset",
        default="./train_b.tsv",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--ds_valid_path",
        help="Valid dataset",
        default="./dev_b.tsv",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--output_folder",
        help="Output folder",
        default="results_taskb",
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
        default=6,
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

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Set seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    ## Parameters
    ds_train_path = args.ds_train_path  # e.g., './train_b.tsv'
    ds_valid_path = args.ds_valid_path  # e.g., './dev_b.tsv'
    output_dir = args.output_folder     # e.g., 'results_taskb/'

    BSZ = args.batch                    # e.g., 8
    EPOCHS = args.num_epochs            # e.g., 5
    LR = args.lr                        # e.g., 5e-5 
    WEIGHT_DECAY = args.weight_decay    # e.g., 0.01
    WARMUP_RATIO = args.warmup_ratio    # e.g., 0.06
    model_path = args.model             # e.g., dbmdz/bert-base-italian-uncased

    ## Output folder
    os.makedirs(output_dir, exist_ok=True)

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2,
        ignore_mismatched_sizes=True)

    ## Train dataset
    train_ds = RegionDataset(
        ds_train_path, 
        model_path, 
        split="train")

    ## Dev dataset
    dev_ds = RegionDataset(
        ds_valid_path, 
        model_path,
        split="dev")

    ## Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
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
        save_total_limit=2)

    ## Data collator
    data_collator = DefaultDataCollator()

    ## Trainer
    trainer = RegressionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=compute_metrics,
        data_collator=data_collator)

    ## Training and evaluation
    trainer.train()
    trainer.save_model(output_dir)
    trainer.evaluate()