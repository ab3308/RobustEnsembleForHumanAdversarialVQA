import torch.cuda
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tarfile
from transformers import BertTokenizer
import mmf
import Inference_extended as Inf
import os

visualbert_gqa_path = "D://VisualBert_GQA//"
visualbert_vizwiz_path = "D://VisualBert_VizWiz//"
visualbert_coco_path = "D://VisualBert_coco"
c_path = "D://Classifiers//25k_each_alternating_capitalised"
labels = ["text", "count", "general"]

model_args = ClassificationArgs(
    use_multiprocessing=False,
    use_multiprocessing_for_evaluation=False,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class EnsembleInference:
    def __init__(self, base_ckpt: str = None, text_ckpt: str = None, logic_ckpt: str = None,
                 classifier_path: str = None):
        self.base_model = Inf.Inference(base_ckpt)
        self.text_model = Inf.Inference(text_ckpt)
        self.logic_model = Inf.Inference(logic_ckpt)
        self.classifier = ClassificationModel(
            "bert", classifier_path,
            num_labels=3,
            args=model_args,
            use_cuda=torch.cuda.is_available()
        )

    def forward(self, image_path: str = None, question: str = None):
        classifier_outputs = self.classifier.predict([question])[0]
        classification = labels[classifier_outputs[0]]
        if classification == "general":
            print("General")
            self.base_model.forward(image_path, question)
        elif classification == "text":
            print("Text")
            self.text_model.forward(image_path, question)
        else:
            print("Logic")
            self.logic_model.forward(image_path, question)


ensemble = EnsembleInference(visualbert_gqa_path, visualbert_gqa_path, visualbert_gqa_path, c_path)
print(ensemble.forward("D://images//gqa_sample_1.png", "How many objects are there?"))
print(ensemble.forward("D://images//gqa_sample_1.png", "What does the book cover say?"))
print(ensemble.forward("D://images//gqa_sample_1.png", "Is there a person in the image?"))

