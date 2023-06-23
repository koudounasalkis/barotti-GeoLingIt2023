"""
python run_train_taska.py \
    --ds_train_path ./train_a.tsv \
    --ds_valid_path ./dev_a.tsv \
    --output_folder results \
    --batch 8 \
    --num_epochs 6 \
    --lr 5e-5 \
    --model dbmdz/bert-base-italian-uncased \
    --seed 42 \
    --load_pretrained_model=True \
    --pretrained_model_path pretraining_contrastive/checkpoint-840
"""



import os

import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers import TrainingArguments, DefaultDataCollator, Trainer
from sklearn.preprocessing import LabelEncoder
from utils import compute_metrics, RegionDataset
from argparse import ArgumentParser


if __name__ == "__main__":

    parser = ArgumentParser(description="EvalITA - GeoLingit")
    parser.add_argument(
        "--ds_train_path",
        help="Train dataset",
        default="./train_a.tsv",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--ds_valid_path",
        help="Valid dataset",
        default="./dev_a.tsv",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--output_folder",
        help="Output folder",
        default="results_taska",
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
        default=10,
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
        "--seed",
        help="Seed Number",
        default=42,
        required=False,
        type=int,
    )
    parser.add_argument(
        "--model",
        help="Model to use",
        default="dbmdz/bert-base-italian-uncased",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--load_pretrained_model",
        help="Load pretrained model",
        default=False,
        required=False,
        type=bool,
    )
    parser.add_argument(
        "--pretrained_model_path",
        help="Path to pretrained model",
        default="pretraining_contrastive/checkpoint-840",
        required=False,
        type=str,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    seed = args.seed

    ## Set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    ## Parameters
    ds_train_path = args.ds_train_path  # e.g., './train_a.tsv'
    ds_valid_path = args.ds_valid_path  # e.g., './dev_a.tsv'
    output_dir = args.output_folder     # e.g., 'results_taska/'

    BSZ = args.batch                    # e.g., 8
    EPOCHS = args.num_epochs            # e.g., 5
    LR = args.lr                        # e.g., 5e-5 
    WEIGHT_DECAY = args.weight_decay    # e.g., 0.01
    WARMUP_RATIO = args.warmup_ratio    # e.g., 0.06
    model_path = args.model             # e.g., dbmdz/bert-base-italian-uncased

    ## Output folder
    output_folder = f"{output_dir}/{model_path}"
    os.makedirs(output_folder, exist_ok=True)

    labels = pd.read_csv(ds_train_path, sep="\t")['region']
    le = LabelEncoder()
    le.fit(labels)
    num_labels = len(list(le.classes_))

    ## Model
    if args.load_pretrained_model:
        print("Loading pretrained model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            args.pretrained_model_path,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            local_files_only=True)
    else:
        print("Loading model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_labels,
            ignore_mismatched_sizes=True)

    ## Train dataset
    train_ds = RegionDataset(
        ds_train_path, 
        model_path, 
        le, 
        split="train")

    ## Dev dataset
    dev_ds = RegionDataset(
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
        metric_for_best_model="f1_macro",
        save_total_limit=2)

    ## Data collator
    data_collator = DefaultDataCollator()

    ## Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=compute_metrics,
        data_collator=data_collator)

    ## Training and evaluation
    trainer.train()
    trainer.save_model(output_folder)
    trainer.evaluate()