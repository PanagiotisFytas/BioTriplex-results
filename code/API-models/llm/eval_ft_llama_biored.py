import json
import re
from copy import deepcopy
import os

system = "hpc"

if system == "hpc":
    output_dir = "/home/pf376/rds/hpc-work/llama-models/metrics/"
    input_dir = "/home/pf376/rds/hpc-work/llama-models/bio_infer/"
else:
    output_dir = f"/mnt/nas_home/pf376/Documents/llama-rec/"
    input_dir = f"/mnt/nas_home/pf376/Documents/llama-rec/"


for mode in ["test", "val", "train"]:
    suffix = f"mode_{mode}_outputs.json"
    if system == "hpc":
        gold_file = f"/home/pf376/biotriplex/data/biored/{mode}_gold_rel_biored.txt"
    else:
        gold_file = f"/mnt/nas_home/pf376/Documents/biotriplex/data/biored/{mode}_gold_rel_biored.txt"
    print("#######################################################")
    print("#######################################################")
    print(f"################# Mode: {mode} #######################")
    print("#######################################################")
    print("#######################################################")
    for output_file in os.listdir(input_dir):
        if not output_file.endswith(suffix):
            continue
        if "_biored_" not in output_file and "_biored_qa_" in output_file:
            continue
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"!!!!!!!!!!!!!!!!!! File: {output_file} !!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        prefix = output_file[:-len(suffix)]

        with open(input_dir + "/" + output_file, "r") as f:
            outputs = json.load(f)

        with open(gold_file, "r") as f:
            gold = f.readlines()

        gold = {(item["doc_key"]): json.loads(item["output"]) for item in [json.loads(line) for line in gold]}
        # gold = {key: [[triplet["gene"], triplet["disease"], triplet["entity_type"]] for triplet in gold[key]] for key in gold}
        gold = {key: [[triplet["gene"], triplet["disease"], triplet["relation"]] for triplet in gold[key]] for key in gold}

        # deduplicate gold for sentence keys
        for key in gold:
            gold[key] = list(set(tuple(triplet) for triplet in gold[key]))
            gold[key] = [list(triplet) for triplet in gold[key]]

        # gold_sentence_keys = {}
        # for key in gold:
        #     for idx, triplet in enumerate(gold[key]):
        #         gold_sentence_keys[key + f"_sentence_{idx}"] = triplet
        #
        # gold = gold_sentence_keys


        dict_outputs ={}
        for doc_key, output in outputs.items():
            # assert doc_key.endswith("_sentence_0"), f"doc_key does not end with '_sentence_0': {doc_key}"
            # Extract the string after "### Response:\n"
            try:
                output = output.split("### Response:\n")[1]
            except IndexError:
                # force empty output
                output = "[]"
            # Convert the string to a list of triplets
            try:
                output = json.loads(output)
            except json.JSONDecodeError:
                print(f"Error in decoding JSON: {output}")
                output = []
                continue
            # Get the true triplets
            try:
                # print(output)
                if output is None or type(output) == int:
                    dict_outputs[doc_key] = []
                else:
                    try:
                        dict_outputs[doc_key] = [[triplet["gene"], triplet["disease"], triplet["relation"]] for triplet in output if (triplet is not None) and (len(triplet) == 3)]
                    except KeyError:
                        dict_outputs[doc_key] = [[triplet["gene"], triplet["disease"], triplet["entity_type"]] for triplet in output if triplet is not None and len(triplet) == 3]
            # print(dict_outputs[doc_key])
            # print(gold[doc_key])
            except KeyError:
                print(f"Error in decoding JSON: {output}")
                output = []
                continue

        # exact match

        # deduplicate outputs for sentence keys
        for key in dict_outputs:
            dict_outputs[key] = list(set(tuple(triplet) for triplet in dict_outputs[key]))
            dict_outputs[key] = [list(triplet) for triplet in dict_outputs[key]]

        TP = 0
        FP = 0
        FN = 0
        TN = 0


        for doc_key in dict_outputs:
            output = dict_outputs[doc_key]
            gold_triplet = gold[doc_key]
            # print(output)
            # print(gold_triplet)
            output_left = deepcopy(output)
            for triplet in gold_triplet:
                if triplet in output_left:
                    TP += 1
                    output_left.remove(triplet)
                else:
                    FN += 1
            FP += len(output_left)

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        print("Exact match")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1: {f1}")

        # partial match

        def eq_triplet(triplet1, triplet2):
            return triplet1[0] == triplet2[0] and triplet1[1] == triplet2[1] and (triplet1[2] in triplet2[2] or triplet2[2] in triplet1[2])

        TP = 0
        FP = 0
        FN = 0
        TN = 0

        for doc_key in dict_outputs:
            output = dict_outputs[doc_key]
            gold_triplet = gold[doc_key]
            # print(output)
            # print(gold_triplet)
            output_left = deepcopy(output)
            for triplet in gold_triplet:
                if triplet in output_left:
                    TP += 1
                    output_left.remove(triplet)
                elif any(eq_triplet(triplet, output_triplet) for output_triplet in output_left):
                    TP += 1
                    for output_triplet in output_left:
                        if eq_triplet(triplet, output_triplet):
                            output_left.remove(output_triplet)
                            break
                else:
                    FN += 1
            FP += len(output_left)

        partial_precision = TP / (TP + FP)
        partial_recall = TP / (TP + FN)
        if partial_precision + partial_recall == 0:
            partial_f1 = 0
        else:
            partial_f1 = 2 * partial_precision * partial_recall / (partial_precision + partial_recall)
        print("Partial match")
        print(f"Precision: {partial_precision}")
        print(f"Recall: {partial_recall}")
        print(f"F1: {partial_f1}")


        # Fully Partial Match

        def full_partial_match(triplet1, triplet2):
            return (triplet1[0] in triplet2[0] or triplet2[0] in triplet1[0]) and (triplet1[1] in triplet2[1] or triplet2[1] in triplet1[1]) and (triplet1[2] in triplet2[2] or triplet2[2] in triplet1[2])

        TP = 0
        FP = 0
        FN = 0
        TN = 0

        for doc_key in dict_outputs:
            output = dict_outputs[doc_key]
            gold_triplet = gold[doc_key]
            # print(output)
            # print(gold_triplet)
            output_left = deepcopy(output)
            for triplet in gold_triplet:
                if triplet in output_left:
                    TP += 1
                    output_left.remove(triplet)
                elif any(eq_triplet(triplet, output_triplet) for output_triplet in output_left):
                    TP += 1
                    for output_triplet in output_left:
                        if eq_triplet(triplet, output_triplet):
                            output_left.remove(output_triplet)
                            break
                elif any(full_partial_match(triplet, output_triplet) for output_triplet in output_left):
                    TP += 1
                    for output_triplet in output_left:
                        if full_partial_match(triplet, output_triplet):
                            output_left.remove(output_triplet)
                            break
                else:
                    FN += 1
            FP += len(output_left)


        full_partial_precision = TP / (TP + FP)
        full_partial_recall = TP / (TP + FN)
        if full_partial_precision + full_partial_recall == 0:
            full_partial_f1 = 0
        else:
            full_partial_f1 = 2 * full_partial_precision * full_partial_recall / (full_partial_precision + full_partial_recall)
        print("Full Partial match")
        print(f"Precision: {full_partial_precision}")
        print(f"Recall: {full_partial_recall}")
        print(f"F1: {full_partial_f1}")

        # excluding relations
        TP = 0
        FP = 0
        FN = 0
        TN = 0

        for doc_key in dict_outputs:
            output = dict_outputs[doc_key]
            gold_triplet = gold[doc_key]
            # print(output)
            # print(gold_triplet)
            output_left = deepcopy(output)
            for triplet in gold_triplet:
                if triplet[:2] in [output_triplet[:2] for output_triplet in output_left]:
                    TP += 1
                    for output_triplet in output_left:
                        if output_triplet[:2] == triplet[:2]:
                            output_left.remove(output_triplet)
                            break
                else:
                    FN += 1
            FP += len(output_left)

        gd_precision = TP / (TP + FP)
        gd_recall = TP / (TP + FN)
        if gd_precision + gd_recall == 0:
            gd_f1 = 0
        else:
            gd_f1 = 2 * gd_precision * gd_recall / (gd_precision + gd_recall)

        print("Excluding relations")
        print(f"Precision: {gd_precision}")
        print(f"Recall: {gd_recall}")
        print(f"F1: {gd_f1}")

        # save the results into a text file
        with open(f"{output_dir}{prefix}mode_{mode}_results.txt", "w") as f:
            f.write("Exact match\n")
            f.write(f"Precision: {precision}\n")
            f.write(f"Recall: {recall}\n")
            f.write(f"F1: {f1}\n")
            f.write("Partial match\n")
            f.write(f"Precision: {partial_precision}\n")
            f.write(f"Recall: {partial_recall}\n")
            f.write(f"F1: {partial_f1}\n")
            f.write("Output\n")
            f.write("Full Partial match\n")
            f.write(f"Precision: {full_partial_precision}\n")
            f.write(f"Recall: {full_partial_recall}\n")
            f.write(f"F1: {full_partial_f1}\n")
            f.write("Excluding relations\n")
            f.write(f"Precision: {gd_precision}\n")
            f.write(f"Recall: {gd_recall}\n")
            f.write(f"F1: {gd_f1}\n")
            f.write("Output\n")
