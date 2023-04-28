import numpy as np
import json
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import vqa_accuracy as accuracy
import numpy as np
from sklearn.model_selection import train_test_split
import os
import openpyxl
import tarfile
# import predict_model
import nltk

vqa_path = "D://vqa_v1_train//OpenEnded_abstract_v002_train2015_questions.json"
vqa_v2_path = "D://vqa_v2_train//v2_OpenEnded_mscoco_train2014_questions.json"

clevr_path = "D://CLEVR_v1.0_no_images//CLEVR_v1.0//questions//CLEVR_train_questions.json"
gqa_path = "D://gqa_questions//test_all_questions.json"

textvqa_path = "D://textvqa_train_questions.json"
ocr_path = "D://ocr-vqa-200k.json"
vizwiz_path = "D://classifier_data//vizwiz_annotations//val.json"

advqa_annotation_path = "D://advqa_annot.json"
advqa_questions_path = "D://advqa_q.json"

advqa_prediction_path = "D://predictions.json"

def get_advqa_vocab():
    data = extract_json(advqa_annotation_path)
    set = {ans for annotation in data for answers in annotation["answers"] for ans in answers["answer"].split(" ")}
    with open("D://advqa_vocab.txt", mode="w", encoding="utf-8") as f:
        [f.write("\n"+answer.lower()) for answer in set]
    f.close()
    print("done")


def extract_tar(source_address, destination_address):
    file = tarfile.open(source_address)
    file.extractall(destination_address)
    file.close()


def extract_npy(file_path):
    return np.load(file_path, allow_pickle=True)[1:]


def extract_json(file_path):
    """
    :param file_path: pass in path to file used
    :return: array of questions for all datasets excluding advqa
    """
    data = json.load(open(file_path))
    if file_path == gqa_path:
        return np.array(list(question['question'] for question in data.values()), dtype=object)
    if file_path == ocr_path:
        questions = np.array(list(question['questions'] for question in data.values()), dtype=object)
        return join_datasets(questions)
    elif file_path == textvqa_path:
        return np.array(list(question['question'] for question in data['data']), dtype=object)
    elif file_path == vizwiz_path:
        return np.array(list(question['question'] for question in data), dtype=object)
    elif file_path == advqa_annotation_path:
        #dict of question id - answers mappings
        return {annotation["question_id"]: annotation["answers"] for annotation in data["annotations"]}
    elif file_path == advqa_questions_path:
        return np.array(list(question for question in data["questions"]))
    elif file_path == advqa_prediction_path:
        return np.array(list(result for result in data))
    else:
        return np.array(list(question['question'] for question in data['questions']), dtype=object)


def rejoin_tokenized_questions(dataset):
    for entry in range(len(dataset)):
        dataset[entry] = " ".join(dataset[entry])
    return dataset


def join_datasets(dataset_list):
    return np.array([question for dataset in dataset_list for question in dataset])


def export_dataset(dataset, labels, export_name):
    data_df = pd.DataFrame(dataset, columns=["questions"])
    labels_df = pd.DataFrame(labels, columns=["labels"])
    dataframe = pd.concat([data_df, labels_df], axis=1)
    dataframe.to_excel(export_name, index=False)


def modify_all_lowercase(array):
    return [str[0].lower()+str[1:] for str in array]


def modify_alternating_case(array):
    return [str[0].upper() + str[1:] if index % 2 == 0 else str[0].lower() + str[1:] for index, str in enumerate(array)]


def capitalise_nouns(string):
    tokens = nltk.word_tokenize(string)
    tags = nltk.pos_tag(tokens)
    return " ".join([w.capitalize() if t == "NN" else w for w, t in tags])


def modify_capitalise_nouns(array):
    return [capitalise_nouns(str) for str in array]



def main():

    data = extract_json(advqa_prediction_path)
    answers = extract_json(advqa_annotation_path)
    eval = accuracy.VQAEval()
    accuracies = []
    for prediction in data:
        q_id = prediction["result"]["question_id"]
        ans = prediction["result"]["answer"]
        exp_ans = [a["answer"] for a in answers[q_id]]
        print(f"Expected answers: {exp_ans}")
        print(f"Prediction: {ans}")
        accuracies.append(eval.__call__(gt=exp_ans, res=ans))
    total_acc = float(sum(accuracies)) / len(accuracies)
    print(total_acc)
    # print(extract_npy("D://.cache//torch//mmf//data//datasets//textvqa//defaults//annotations//imdb_train_ocr_en.npy")[0])
    # dataset = [extract_json(vqa_path)[0:25000], extract_json(vqa_v2_path)[0:25000], extract_json(textvqa_path)[0:25000], extract_json(ocr_path)[0:25000], extract_json(clevr_path)[0:25000], extract_json(gqa_path)[0:25000]]
    # dataset = join_datasets(dataset)
    # count = 50000
    # text_labels = ["text"] * count
    # count_labels = ["count"] * count
    # general_labels = ["general"] * count
    #
    # labels = [general_labels, text_labels, count_labels]
    # labels = join_datasets(labels)
    #
    # #export_dataset(dataset, labels, "D://Classifier_Dataset//25k_each.xlsx")
    #
    # alternating_dataset = modify_alternating_case(dataset)
    # export_dataset(alternating_dataset, labels, "D://Classifier_Dataset//25k_each_alternating_case.xlsx")
    #
    # #lowercase_dataset = modify_all_lowercase(dataset)
    # #export_dataset(lowercase_dataset, labels, "D://Classifier_Dataset//25k_each_lowercase.xlsx")
    #
    # capitalised_alternating = modify_capitalise_nouns(alternating_dataset)
    # export_dataset(capitalised_alternating, labels, "D://Classifier_Dataset//25k_each_alternating_case_capitalised_nouns.xlsx")
    #
    # #capitalised_lower = modify_capitalise_nouns(lowercase_dataset)
    # #export_dataset(capitalised_lower, labels, "D://Classifier_Dataset//25k_each_lowercase_capitalised_nouns.xlsx")

if __name__ == "__main__":
    main()
