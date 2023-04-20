import torch.cuda
from simpletransformers.classification import ClassificationModel as cm
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import tarfile
import simpletransformers


def f1_multiclass(labels, preds):
    return f1_score(labels, preds, average="micro")


def main():
    labels = ["text", "count", "general"]
    dataset_df = pd.read_excel("./Classifier_Dataset/25k_each_lowercase_capitalised_nouns.xlsx")
    dataset_df["labels"] = dataset_df.apply(lambda x: labels.index(x["labels"]), axis=1)
    train_df, test_df = train_test_split(dataset_df, test_size=0.1)

    train_args = {
        "reprocess_input_data": True,
        "fp16": True,
        "num_train_epochs": 60,
        "overwrite_output_dir": True,
        "best_model_dir": "./text_classifier/25k_each_lowercase_capitalised/",
        "evaluate_during_training": True,
        "evaluate_during_training_verbose": True,

    }

    model_args = simpletransformers.classification.ClassificationArgs(
        use_multiprocessing=False,
        use_multiprocessing_for_evaluation=False,
        fp16=True,
        num_train_epochs=25,
        overwrite_output_dir=True,
        best_model_dir="./text_classifier/25k_each_lowercase_capitalised/",
        output_dir="./text_classifier/25k_each_lowercase_capitalised_best/",
        evaluate_each_epoch=True,
        evaluate_during_training=True,
        evaluate_during_training_verbose=True,
        use_early_stopping=True,
        early_stopping_patience=5,
        early_stopping_delta=0.0001
    )

    model = cm(
        "bert", "prajjwal1/bert-tiny",
        num_labels=3,
        args=model_args,

    )

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print("created model, beginning training")
    model.train_model(train_df=train_df, eval_df=test_df)

    result, model_outputs, wrong_predictions = model.eval_model(test_df, f1=f1_multiclass, acc=accuracy_score)
    print(result)


if __name__ == "__main__":
    main()
