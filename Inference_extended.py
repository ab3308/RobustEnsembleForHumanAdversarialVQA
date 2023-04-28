import torch
import json
import torchvision.models as models
import torchvision.transforms as transforms

import dataset_manipulation as datasets
from transformers import BertTokenizer

from mmf.common.report import Report
from mmf.common.sample import Sample, SampleList
from mmf.utils.build import build_model, build_processors
from mmf.utils.checkpoint import load_pretrained_model
from mmf.utils.general import get_current_device
from omegaconf import OmegaConf
from PIL import Image


transform = transforms.Compose([
    transforms.Resize((448, 448)),  # was 224 but 448 to be consistent with original
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



class Inference:
    """
    Adaptation of mmf/utils/inference.py, for VisualBERT image-by-image inference
    """

    def __init__(self, checkpoint_path: str = None):
        self.checkpoint = checkpoint_path
        assert self.checkpoint is not None
        self.processor, self.model = self._build_model()
        resnet152 = models.resnet152(pretrained=True)
        self.resnet152 = torch.nn.Sequential(*(list(resnet152.children())[:-1])).eval()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def extract_resnet152(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)  # individual image passed, expects batch
        # extract features
        with torch.no_grad():
            # disable gradient calculation - unnecessary - only doing forward pass
            features = self.resnet152(image_tensor).squeeze()
        return features

    def process_question(self, question):
        token = self.tokenizer.tokenize(question)
        input_ids = self.tokenizer.convert_tokens_to_ids(token)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)
        lm_label_ids = [-1] * len(input_ids)
        return token, input_ids, input_mask, segment_ids, lm_label_ids

    def get_mmf_sample(self, image_path, question):
        features = self.extract_resnet152(image_path)
        num_regions = features.size(0)
        image_dim = torch.tensor([num_regions])
        tokens, input_ids, input_mask, segment_ids, lm_label_ids = self.process_question(question)

        sample = Sample({
            "text": tokens,
            "input_ids": torch.tensor(input_ids),  # convert tokens to ids via tokenizer
            "input_mask": torch.tensor(input_mask),  # 1 = text inp, 0 = img features (when concatenated)
            "segment_ids": torch.tensor(segment_ids),  #
            "lm_label_ids": torch.tensor(lm_label_ids),
            "image_feature_0": features,
            "image_dim": image_dim,
        })
        return sample

    def _build_model(self):
        self.model_items = load_pretrained_model(self.checkpoint)
        self.config = OmegaConf.create(self.model_items["full_config"])
        dataset_name = list(self.config.dataset_config.keys())[0]
        processor = build_processors(
            self.config.dataset_config[dataset_name].processors
        )
        ckpt = self.model_items["checkpoint"]
        model = build_model(self.model_items["config"])
        model.load_state_dict(ckpt)
        return processor, model

    def output_advqa_predictions(self):
        model = Inference("D://best//")

        questions = datasets.extract_json(datasets.advqa_questions_path)
        results = []
        for question in questions:
            img_id = question["image_id"]
            q = question["question"]
            img_path = "D://advqa_images//" + img_id + ".jpg"
            answer = model.forward(img_path, q)

            data = {
                "result": {
                    "question_id": question["question_id"],
                    "answer": answer
                }
            }
            results.append(data)

        with open("./advqa_predictions.json", mode="w", encoding="utf-8") as f:
            json.dump(results, f)
        f.close()
        # annotations = dataset_manipulation.extract_json(dataset_manipulation.advqa_annotation_path)

    def get_advqa_score(self):
        predictions = datasets.extract_json(datasets.advqa_prediction_path)
        answers = datasets.extract_json(datasets.advqa_annotation_path)
        for prediction in predictions:
            count = 0
            q_id = prediction["question_id"]
            predicted_ans = prediction["answer"]
            expected_ans = answers[q_id]
            for answer in expected_ans:
                if answer["answer"].lower() == predicted_ans.lower():
                    count += 1


    def forward(self, path: str = None, question: str = None):
        sample = self.get_mmf_sample(path, question)

        sample_list = SampleList([sample])
        sample_list = sample_list.to(get_current_device())
        self.model = self.model.to(get_current_device())
        output = self.model(sample_list)
        sample_list.id = [sample_list.input_ids[0][0]]
        report = Report(sample_list, output)

        answers = output["scores"].argmax(dim=-1)
        answer_str = self.processor["answer_processor"].idx2word(answers.item())
        report["answers"] = answer_str
        return report["answers"]
# def main():
#     inference = Inference("D://best//")
#     inference.output_advqa_predictions()
#
# if __name__ == "__main__":
#     main()
