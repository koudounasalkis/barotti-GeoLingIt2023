import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

import argparse

""" 
    This code computes the entropy-based ensemble of the two models,
    namely LR and BERT.
"""
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="EvalITA - GeoLingit")
    parser.add_argument(
        "--dev_lr_confidence",
        help="Dev LR confidence scores",
        default="./dev_lr_confidence.tsv",
        required=False,
        type=str,)
    parser.add_argument(
        "--dev_bert_confidence",
        help="Dev BERT confidence scores",
        default="./dev_bert_confidence.tsv",
        required=False,
        type=str,)
    parser.add_argument(
        "--train_file",
        help="Path to train file",
        default="train_a.tsv",
        required=False,
        type=str,
        )
    args = parser.parse_args()

    ## Load confidence scores
    results_lr = pd.read_csv(args.dev_lr_confidence, delimiter='\t', index_col=None)
    results_lr = results_lr.drop(columns=['id'])
    results_bert = pd.read_csv(args.dev_bert_confidence, delimiter='\t', index_col=None)
    regions = [c for c in results_lr.columns if c not in ["id", "text", "region"]]

    ## Get label names
    le = LabelEncoder()
    labels = pd.read_csv(args.train_file, sep="\t")['region'].values.tolist()
    le.fit(labels)
    labels = le.transform(results_bert['region'].values.tolist())

    ## Check predictions and metrics
    prediction_lr = results_lr[regions].values.argmax(axis=1)
    print("F1 Macro: ", round(f1_score(labels, prediction_lr, average="macro"),4))
    print("F1 Weighted: ", round(f1_score(labels, prediction_lr, average="weighted"),4))
    print("Accuracy: ", round(accuracy_score(labels, prediction_lr),4))
    print("----------------\n")

    prediction_bert = results_bert[regions].values.argmax(axis=1)
    print("F1 Macro: ", round(f1_score(labels, prediction_bert, average="macro"),4))
    print("F1 Weighted: ", round(f1_score(labels, prediction_bert, average="weighted"),4))
    print("Accuracy: ", round(accuracy_score(labels, prediction_bert),4))
    print("----------------\n")

    ## Ensamble
    prediction_proba_lr = results_lr[regions].values
    prediction_proba_bert = results_bert[regions].values
    threshold_entropy = list(np.arange(0.5, 2.0, 0.1))
    for te in threshold_entropy:
        final_predictions_entropy = []
        for proba_lr, proba_bert, pred in zip(prediction_proba_lr, prediction_proba_bert, predictions):
            entropy_lr = np.sum(- proba_lr * np.log(proba_lr))
            entropy_bert = np.sum(- proba_bert * np.log(proba_bert))
            entropy = np.argmin(np.array([entropy_lr, entropy_bert]))
            if entropy == 1:
                if entropy_bert < te:
                    final_predictions_entropy.append(pred[entropy])
                else:
                    final_predictions_entropy.append(pred[0])
            else:
                final_predictions_entropy.append(pred[entropy])

        print("----------------")
        print("Threshold ", te)
        print("F1 Macro: ", round(f1_score(labels, final_predictions_entropy, average="macro"),4))
        print("F1 Weighted: ", round(f1_score(labels, final_predictions_entropy, average="weighted"),4))
        print("Accuracy: ", round(accuracy_score(labels, final_predictions_entropy),4))
        print("----------------\n")