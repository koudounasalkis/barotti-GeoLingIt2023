"""
python run_inference_taska.py \
    --ds_train_path ./train_a.tsv \
    --ds_valid_path ./dev_a.tsv \
    --output_folder results_taska/ \
"""

import os

import pandas as pd
import torch
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
        default="results_taska/",
        required=False,
        type=str,
        )
    parser.add_argument(
        "--ckpt_to_evaluate",
        help="Model ckpt to evaluate",
        default="best-model-ckpt/",
        required=True,
        type=str,
        )
    parser.add_argument(
        "--model",
        help="Model name",
        default="dbmdz/bert-base-italian-uncased",
        required=True,
        type=str,
        )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Parameters
    ds_train_path = args.ds_train_path  # e.g., './train_a.tsv'
    ds_valid_path = args.ds_valid_path  # e.g., './dev_a.tsv'
    output_dir = args.output_folder     # e.g., 'inference/'
    ckpt_to_evaluate = args.ckpt_to_evaluate
    model = args.model

    BSZ = 8
    LR = 5e-5                      
    WEIGHT_DECAY = 0.01    
    WARMUP_RATIO = 0.06    
           
    print(f"Model to evaluate: {ckpt_to_evaluate}")
    print(f"Tokenizer: {model}")

    ## Output folder
    os.makedirs(output_dir, exist_ok=True)

    labels = pd.read_csv(ds_train_path, sep="\t")['region']
    le = LabelEncoder()
    le.fit(labels)
    num_labels = len(list(le.classes_))

    ## Model
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt_to_evaluate,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        local_files_only=True)
    print("Model loaded!\n")

    ## Train dataset
    print("Loading datasets...")
    train_ds = RegionDataset(
        ds_train_path, 
        model, 
        le, 
        split="train")

    ## Dev dataset
    dev_ds = RegionDataset(
        ds_valid_path, 
        model, 
        le, 
        split="dev")

    print("Datasets loaded!\n")

    ## Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BSZ,
        per_device_eval_batch_size=BSZ
        )

    ## Data collator
    data_collator = DefaultDataCollator()

    ## Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=dev_ds,
        compute_metrics=compute_metrics,
        data_collator=data_collator)

    ## Predictions
    predictions = trainer.predict(dev_ds)
    print(predictions.metrics)

    ## Compute confidence scores
    sof = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=-1)
    df = pd.DataFrame(sof, columns=le.classes_)
    df["region"] = le.inverse_transform(predictions.label_ids) 
    df["index"] = 
    df.to_csv(f"{output_dir}/dev_bert_confidence.tsv", sep="\t", index=False)

    ## Save predictions
    predictions = predictions.predictions.argmax(axis=1)
    predictions = le.inverse_transform(predictions)
    with open(f"{output_dir}/dev_predictions.csv", "w") as f:
        f.write("id,region\n")
        for i, pred in enumerate(predictions):
            f.write(f"{i+13670},{pred}\n")   
