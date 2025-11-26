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


def parse_output(output):
    """
    :param output: in the form of "word: tag\n..."
    :return: a dictionary of the form {"tag": ["word1", "word2", ...], ...}
    """
    output = output.split("\n")
    output = [line.split(": ") for line in output if line]
    output = {tag: [word for word, tag_ in output if tag_ == tag] for word, tag in output}
    # make tags from GENE, DISEASE, RELATION, NULL to genes, diseases, relations, null
    for tag in list(output.keys()):
        if "GENE" in tag:
            output["genes"] = output.pop(tag)
        elif "DISEASE" in tag:
            output["diseases"] = output.pop(tag)
        elif "RELATION" in tag:
            output["relations"] = output.pop(tag)
        elif "NULL" in tag:
            output["null"] = output.pop(tag)
    # if any of the genes, diseases, relations, null is not in the output, add an empty list
    for entity_type in ["genes", "diseases", "relations", "null"]:
        if entity_type not in output:
            output[entity_type] = []
    return output

def parse_output_with_errors(output, newline_sep=True):
    """
    :param output: in the form of "word: tag\n...". If there is an error at a line, skip it
    :return: a dictionary of the form {"tag": ["word1", "word2", ...], ...}
    """
    if newline_sep:
        output = output.split("\n")
    else:
        # split on ", "
        output = re.split(", ", output)
    output = [line.split(": ") for line in output if line]
    # skip lines that do not have an item before and after ": ", and that the second item is not in ["GENE", "DISEASE", "RELATION", "NULL"]
    output_fix = [line for line in output if len(line) == 2 and line[1] in ["GENE", "DISEASE", "RELATION", "NULL"]]
    if len(output) != len(output_fix):
        print(f"Skipped {len(output) - len(output_fix)} lines!!!!!!!!!!!!!!!!!!!")
    output = output_fix
    output = {tag: [word for word, tag_ in output if tag_ == tag] for word, tag in output}
    # make tags from GENE, DISEASE, RELATION, NULL to
    # print(output)
    for tag in list(output.keys()):
        if "GENE" in tag:
            output["genes"] = output.pop(tag)
        elif "DISEASE" in tag:
            output["diseases"] = output.pop(tag)
        elif "RELATION" in tag:
            output["relations"] = output.pop(tag)
        elif "NULL" in tag:
            output["null"] = output.pop(tag)
    # if any of the genes, diseases, relations, null is not in the output, add an empty list
    for entity_type in ["genes", "diseases", "relations", "null"]:
        if entity_type not in output:
            output[entity_type] = []
    return output

for mode in ["test",]:# "val", "train"]:
    suffix = f"mode_{mode}_outputs.json"
    if system == "hpc":
        gold_file = f"/home/pf376/biotriplex/data/{mode}_gold_nerlong.txt"
    else:
        gold_file = f"/mnt/nas_home/pf376/Documents/biotriplex/data/{mode}_gold_nerlong.txt"
    print("#######################################################")
    print("#######################################################")
    print(f"################# Mode: {mode} #######################")
    print("#######################################################")
    print("#######################################################")

    with open(gold_file, "r") as f:
        gold = f.readlines()

    gold = {(item["doc_key"]): parse_output(item["output"]) for item in [json.loads(line) for line in gold]}
    gold = [{"doc_key": key, "output": gold[key]} for key in gold]

    for output_file in os.listdir(input_dir):
        if not output_file.endswith(suffix):
            continue
        if "nerlong" not in output_file:
            continue
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"!!!!!!!!!!!!!!!!!! File: {output_file} !!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        prefix = output_file[:-len(suffix)]

        with open(input_dir + "/" + output_file, "r") as f:
            outputs = json.load(f)

        gold_outputs = {sample["doc_key"]: {
            "genes": sample["output"]["genes"],
            "diseases": sample["output"]["diseases"],
            "relations": sample["output"]["relations"]
        } for sample in gold}

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
                if "shot" in output_file:
                    output = output.split("assistant\n\n")[-1]
                else:
                    output = output.split("### Response:\n")[1]
            except IndexError:
                # force empty output
                output = "[]"
            # Convert the string to a list of dictionaries
            output = parse_output_with_errors(output, newline_sep=("0shot" not in output_file))
            # Get the true triplets
            dict_outputs[doc_key] = {
                "genes": output["genes"],
                "diseases": output["diseases"],
                "relations": output["relations"]
            }
            # print(dict_outputs[doc_key])
            # print(gold[doc_key])


        # exact match

        TP = {
            "genes": 0,
            "diseases": 0,
            "relations": 0
        }
        FP = {
            "genes": 0,
            "diseases": 0,
            "relations": 0
        }
        FN = {
            "genes": 0,
            "diseases": 0,
            "relations": 0
        }
        TN = {
            "genes": 0,
            "diseases": 0,
            "relations": 0
        }


        for doc_key in dict_outputs:
            output = dict_outputs[doc_key]
            gold_output = gold_outputs[doc_key]
            # print(output)
            # print(gold_triplet)
            output_left = deepcopy(output)
            for entity_type in ["genes", "diseases", "relations"]:
                for entity in gold_output[entity_type]:
                    if entity in output_left[entity_type]:
                        TP[entity_type] += 1
                        output_left[entity_type].remove(entity)
                    else:
                        FN[entity_type] += 1
                FP[entity_type] += len(output_left[entity_type])

        precision = {entity_type: TP[entity_type] / (TP[entity_type] + FP[entity_type])  if TP[entity_type] + FP[entity_type] != 0 else 0
                     for entity_type in ["genes", "diseases", "relations"]}
        recall = {entity_type: TP[entity_type] / (TP[entity_type] + FN[entity_type]) if TP[entity_type] + FN[entity_type] != 0 else 0
                  for entity_type in ["genes", "diseases", "relations"]}
        f1 = {"entity_type": 0 for entity_type in ["genes", "diseases", "relations"]}
        for entity_type in ["genes", "diseases", "relations"]:
            if precision[entity_type] + recall[entity_type] == 0:
                f1[entity_type] = 0
            else:
                f1[entity_type] = 2 * precision[entity_type] * recall[entity_type] /\
                                  (precision[entity_type] + recall[entity_type])\
                    if precision[entity_type] + recall[entity_type] != 0 else 0
        print("Exact match")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1: {f1}")
        overall_TP = sum(TP.values())
        overall_FP = sum(FP.values())
        overall_FN = sum(FN.values())
        overall_precision = overall_TP / (overall_TP + overall_FP)\
            if overall_TP + overall_FP != 0 else 0
        overall_recall = overall_TP / (overall_TP + overall_FN)\
            if overall_TP + overall_FN != 0 else 0
        if overall_precision + overall_recall == 0:
            overall_f1 = 0
        else:
            overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall)\
                if overall_precision + overall_recall != 0 else 0
        print("\tOverall")
        print(f"\tPrecision: {overall_precision}")
        print(f"\tRecall: {overall_recall}")
        print(f"\tF1: {overall_f1}")


        # partial match

        def eq_entity(entity1, entity2):
            return (entity1 in entity2) or (entity2 in entity1 ) or (entity1 == entity2)

        TP = {
            "genes": 0,
            "diseases": 0,
            "relations": 0
        }
        FP = {
            "genes": 0,
            "diseases": 0,
            "relations": 0
        }
        FN = {
            "genes": 0,
            "diseases": 0,
            "relations": 0
        }
        TN = {
            "genes": 0,
            "diseases": 0,
            "relations": 0
        }

        for doc_key in dict_outputs:
            output = dict_outputs[doc_key]
            gold_output = gold_outputs[doc_key]
            # print(output)
            # print(gold_triplet)
            output_left = deepcopy(output)
            for entity_type in ["genes", "diseases", "relations"]:
                for entity in gold_output[entity_type]:
                    if entity in output_left[entity_type]:
                        TP[entity_type] += 1
                        output_left[entity_type].remove(entity)
                    elif any(eq_entity(entity, output_entity) for output_entity in output_left[entity_type]):
                        TP[entity_type] += 1
                        for output_entity in output_left[entity_type]:
                            if eq_entity(entity, output_entity):
                                output_left[entity_type].remove(output_entity)
                                break
                    else:
                        FN[entity_type] += 1
                FP[entity_type] += len(output_left[entity_type])

        partial_precision = {entity_type: TP[entity_type] / (TP[entity_type] + FP[entity_type])\
                             if TP[entity_type] + FP[entity_type] != 0 else 0
                             for entity_type in ["genes", "diseases", "relations"]}
        partial_recall = {entity_type: TP[entity_type] / (TP[entity_type] + FN[entity_type])
                          if TP[entity_type] + FN[entity_type] != 0 else 0
                          for entity_type in ["genes", "diseases", "relations"]}
        partial_f1 = {"entity_type": 0 for entity_type in ["genes", "diseases", "relations"]}
        for entity_type in ["genes", "diseases", "relations"]:
            if partial_precision[entity_type] + partial_recall[entity_type] == 0:
                partial_f1[entity_type] = 0
            else:
                partial_f1[entity_type] = 2 * partial_precision[entity_type] * partial_recall[entity_type]\
                                          / (partial_precision[entity_type] + partial_recall[entity_type])\
                    if partial_precision[entity_type] + partial_recall[entity_type] != 0 else 0
        print("Partial match")
        print(f"Precision: {partial_precision}")
        print(f"Recall: {partial_recall}")
        print(f"F1: {partial_f1}")

        overall_TP = sum(TP.values())
        overall_FP = sum(FP.values())
        overall_FN = sum(FN.values())
        overall_partial_precision = overall_TP / (overall_TP + overall_FP)\
            if overall_TP + overall_FP != 0 else 0
        overall_partial_recall = overall_TP / (overall_TP + overall_FN)\
            if overall_TP + overall_FN != 0 else 0
        if overall_partial_precision + overall_partial_recall == 0:
            overall_partial_f1 = 0
        else:
            overall_partial_f1 = 2 * overall_partial_precision * overall_partial_recall /\
                                 (overall_partial_precision + overall_partial_recall)\
                if overall_partial_precision + overall_partial_recall != 0 else 0
        print("\tOverall")
        print(f"\tPrecision: {overall_partial_precision}")
        print(f"\tRecall: {overall_partial_recall}")
        print(f"\tF1: {overall_partial_f1}")

        # save the results into a text file
        with open(f"{output_dir}{prefix}mode_{mode}_results.txt", "w") as f:
            f.write("Exact match\n")
            f.write(f"Precision: {precision}\n")
            f.write(f"Recall: {recall}\n")
            f.write(f"F1: {f1}\n")
            f.write(f"\tOverall\n")
            f.write(f"\tPrecision: {overall_precision}\n")
            f.write(f"\tRecall: {overall_recall}\n")
            f.write(f"\tF1: {overall_f1}\n")
            f.write("Partial match\n")
            f.write(f"Precision: {partial_precision}\n")
            f.write(f"Recall: {partial_recall}\n")
            f.write(f"F1: {partial_f1}\n")
            f.write(f"\tOverall\n")
            f.write(f"\tPrecision: {overall_partial_precision}\n")
            f.write(f"\tRecall: {overall_partial_recall}\n")
            f.write(f"\tF1: {overall_partial_f1}\n")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
