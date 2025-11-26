import torch
from awq import AutoAWQForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from json.decoder import JSONDecodeError
from tqdm import tqdm
from warnings import warn
import fileinput

from prompt import triplet_prompt, system_prompt, sample_input

def wait_for_yes():
    response = input("Do you want to continue? (yes/no): ").strip().lower()
    while response != "yes":
        print("Waiting for 'yes' to continue...")
        response = input("Do you want to continue? (yes/no): ").strip().lower()

    print("Proceeding with the next steps...")

# Example usage


def main():

    MAX_OUTPUT_LEN = 1024
    model_id = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoAWQForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )


    test_set_path = "/mnt/nas_home/pf376/Documents/biotriplex/data/val.txt"
    output_path = "/mnt/nas_home/pf376/Documents/biotriplex/data/val_output.txt"
    # each line is a json object with the following keys:
    # {
    #         "doc_key": "<paper_id>_<section_name>",
    #         "text": [["text1"], ..., ["textM"]],
    #         "ner": [[[char_start>, <char_end>, <entity_type>],...]] # sorted by char_start
    #         "relations": [[[<start_char_1>, <end_char_1>, <start_char_2>, <end_char_2>, <relation label>], ...]]
    #         "triplets": [[[<gene_start>, <gene_end>, <disease_start>, <disease_end>, <relation start>, <relation end>], ...]]
    #         "triplets_text": [[[<gene_text>, <disease_text>, <relation_text>], ...]]
    #         "dataset": "BioTriplEx",
    #         "_split": "train/val/test"
    #         "_sentence_index": <sentence_index>
    #     }
    # read the test set
    with open(test_set_path, "r") as f:
        test_set = f.readlines()
    test_set = [json.loads(line) for line in test_set]

    preds = {}
    for sample in tqdm(test_set):
        preds_per_sample = []
        for idx, sentence in enumerate(sample["sentences"]):
            input_text = triplet_prompt + sentence
            prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text}
            ]
            input = tokenizer.apply_chat_template(
              prompt,
              tokenize=True,
              add_generation_prompt=True,
              return_tensors="pt",
              return_dict=True,
            ).to("cuda")
            output = model.generate(**input,
                                     do_sample=False,
                                     num_beams=1,
                                     max_new_tokens=MAX_OUTPUT_LEN)
            print(tokenizer.batch_decode(input['input_ids'], skip_special_tokens=True)[0])
            print(tokenizer.batch_decode(output[:, input['input_ids'].shape[1]:], skip_special_tokens=True)[0])
            # wait for user input to continue
            if len(tokenizer.batch_decode(output[:, input['input_ids'].shape[1]:], skip_special_tokens=True)[0]) < 4:
                continue
            print("DOc key: ", sample["doc_key"])
            wait_for_yes()

             # print(sample["triplets_text"][idx])
            output_text = tokenizer.batch_decode(output[:, input['input_ids'].shape[1]:], skip_special_tokens=True)[0]
            preds_per_sample.append(output_text)
        # print("Doc key: ", sample["doc_key"])
        preds[sample["doc_key"]] = preds_per_sample

    with open(output_path, "w") as f:
        for key, value in preds.items():
            f.write(json.dumps({"doc_key": key, "sentences": value}) + "\n")
    print("Done!")


def calculate_metrics(output_path=None):
    test_set_path = "/mnt/nas_home/pf376/Documents/biotriplex/data/val.txt"
    if output_path is None:
        output_path = "/mnt/nas_home/pf376/Documents/biotriplex/data/val_output.txt"
    with open(test_set_path, "r") as f:
        test_set = f.readlines()
    test_set = [json.loads(line) for line in test_set]
    with open(output_path, "r") as f:
        output = f.readlines()
    output = [json.loads(line) for line in output]
    for sample in output:
        sample["triplets_per_sentence"] = []
        for sentence_prediction_text in sample["sentences"]:
            try:
                sentence_prediction_list = json.loads(sentence_prediction_text)
            except JSONDecodeError:
                warning_message = "Could not decode the following string: " + sentence_prediction_text
                warn(warning_message)
                # if there are no brackets the string is empty:
                if "{" not in sentence_prediction_text and "}" not in sentence_prediction_text:
                    fixed = "[]"
                else:
                    # get everything until the last closed bracket
                    fixed = sentence_prediction_text[:sentence_prediction_text.rfind("}") + 1] + "]"
                # print("Fixed: ", fixed)
                try:
                    sentence_prediction_list = json.loads(fixed)
                except JSONDecodeError:
                    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    # print("Could not decode the following string: ", fixed)
                    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    sentence_prediction_list = []
            sentence_prediction_list = [[item["Gene"], item["Human Disease"], item["Relation"]] for item in sentence_prediction_list]
            sample["triplets_per_sentence"].append(sentence_prediction_list)
    # print(output)

    # make into a dictionary
    for lowercase in [True, False]:
        for whitespace_remove in [True, False]:
            for ignore_relation in [True, False]:
                TP = 0
                FP = 0
                FN = 0
                TN = 0
                for idx in range(len(test_set)):
                    # print("INDEX: ", idx)
                    # print("Ground truth: ", test_set[idx]["triplets_text"])
                    # print("Prediction: ", output[idx]["triplets_per_sentence"])
                    assert test_set[idx]["doc_key"] == output[idx]["doc_key"]
                    assert len(test_set[idx]["sentences"]) == len(output[idx]["triplets_per_sentence"])
                    for i in range(len(test_set[idx]["sentences"])):
                        # print("Sentence: ", i)
                        ground_truth = test_set[idx]["triplets_text"][i]
                        prediction = output[idx]["triplets_per_sentence"][i]
                        # print("Ground truth: ", ground_truth)
                        # print("Prediction: ", prediction)
                        if lowercase:
                            prediction = [[item.lower() for item in triplet] for triplet in prediction]
                            ground_truth = [[item.lower() for item in triplet] for triplet in ground_truth]
                        if whitespace_remove:
                            prediction = [[item.strip() for item in triplet] for triplet in prediction]
                            ground_truth = [[item.strip() for item in triplet] for triplet in ground_truth]
                        if ignore_relation:
                            prediction = [[item[0], item[1]] for item in prediction]
                            ground_truth = [[item[0], item[1]] for item in ground_truth]
                        for item in prediction:
                            if item in ground_truth:
                                TP += 1
                            else:
                                FP += 1
                        for item in ground_truth:
                            if item not in prediction:
                                FN += 1
                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                f1 = 2 * precision * recall / (precision + recall)
                print("Lowercase: ", lowercase)
                print("Whitespace remove: ", whitespace_remove)
                print("Ignore relation: ", ignore_relation)
                print("TP: ", TP)
                print("FP: ", FP)
                print("FN: ", FN)
                print("Precision: ", precision)
                print("Recall: ", recall)
                print("F1: ", f1)




if __name__ == "__main__":
    main()
    # calculate_metrics()
    # print("Done!")
    # calculate_metrics(output_path="/mnt/nas_home/pf376/Documents/biotriplex/data/val_output_long_output.txt")
    # print("Done!")
