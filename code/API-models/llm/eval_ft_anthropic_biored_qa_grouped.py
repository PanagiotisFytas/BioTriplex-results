import json
import re
from copy import deepcopy
import os
import pandas as pd
import warnings
import numpy as np
# import classification_report
from sklearn.metrics import classification_report
# import confusion_matrix
from sklearn.metrics import confusion_matrix

system = "dtal"

if system == "hpc":
    output_dir = "/mnt/nas_home/pf376/Documents/biotriplex/BioTriplExperiments/"
    input_dir = "/mnt/nas_home/pf376/Documents/biotriplex/BioTriplExperiments/"
else:
    output_dir = f"/mnt/nas_home/pf376/Documents/biotriplex/BioTriplExperiments/"
    input_dir = f"/mnt/nas_home/pf376/Documents/biotriplex/BioTriplExperiments/"

def output_to_relation(output):
    if "a" in output:
        if output != "a)":
            warnings.warn(f"Output {output} is not in the expected format")
        return "Association"
    if "b" in output:
        if output != "b)":
            warnings.warn(f"Output {output} is not in the expected format")
        return "Negative_Correlation"
    if "c" in output:
        if output != "c)":
            warnings.warn(f"Output {output} is not in the expected format")
        return "Positive_Correlation"
    if "d" in output:
        if output != "d)":
            warnings.warn(f"Output {output} is not in the expected format")
        return "No relation"
    raise ValueError(f"Output {output} is not in the expected format")


relations = ["Association", "Negative_Correlation", "Positive_Correlation", "No relation", "Null"]

for mode in ["test", ]:#"val", "train"]:
    suffix = f"_output.jsonl"
    if system == "hpc":
        gold_file = f"/home/pf376/biotriplex/data/biored/{mode}_gold_biored_qa.txt"
    else:
        gold_file = f"/mnt/nas_home/pf376/Documents/biotriplex/data/biored/{mode}_gold_biored_qa.txt"
    print("#######################################################")
    print("#######################################################")
    print(f"################# Mode: {mode} #######################")
    print("#######################################################")
    print("#######################################################")
    for output_file in os.listdir(input_dir):
        if not output_file.endswith(suffix):
            continue
        if ("biored_qa" not in output_file) or (mode not in output_file) or ("claude" not in output_file):
            continue
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"!!!!!!!!!!!!!!!!!! File: {output_file} !!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        prefix = output_file[:-len(suffix)]

        with open(input_dir + "/" + output_file, "r") as f:
            outputs = {item["doc_key"]: item["output"][0] for item in [json.loads(line) for line in f.readlines()]}


        with open(gold_file, "r") as f:
            gold = f.readlines()

        gold = {(item["doc_key"]): item["output"] for item in [json.loads(line) for line in gold]}
        # gold = {key: [[triplet["gene"], triplet["disease"], triplet["entity_type"]] for triplet in gold[key]] for key in gold}
        # gold = {key: gold[key] for key in gold}
        gold = {key: output_to_relation(gold[key]) for key in gold}




        dict_outputs = {}
        for doc_key, output in outputs.items():
            # assert doc_key.endswith("_sentence_0"), f"doc_key does not end with '_sentence_0': {doc_key}"
            # Extract the string after "### Response:\n"
            try:
                if "0shot" in output_file:
                    # print("1: ", output)
                    # remove any text between a ")" and a comma, e.g. "a) text," -> "a)," and "a) text" -> "a)"
                    output = re.sub(r"\) [^,]*,", "),", output)
                    # print("2: ", output)
                    # remove any text after the final parenthesis
                    if ")" not in output:
                        output = output.replace(",", "),").rstrip()
                        output += ")"
                    output = output.split(")")
                    if len(output) > 1:
                        output = output[:-1]
                    else:
                        pass
                    # print("3: ", output)
                    output = ")".join(output)
                    if not output.rstrip().endswith(")"):
                        output = output + ")"
                    # print("4: ", output)
                elif "shot" in output_file:
                    if "### Response:\n" in output:
                        output = output.split("### Response:\n")[-1].strip()
                        if not output.endswith(")"):
                            output = output + ")"
                        print("out", output)
                        print("gold", gold[doc_key])
                    else:
                        pass
                else:
                    raise ValueError(f"Supervised not implemented for openai")
                    output = output.split("### Response:\n")[1].strip()
            except IndexError:
                # force empty output
                output = "Null"
            # Convert the string to a list of triplets
            try:
                output = output_to_relation(output)
            except ValueError:
                print(f"Error in decoding: {output}")
                output = "Null"
                continue
            # Get the true triplets
            except KeyError:
                print(f"Error in output: {output}")
                output = "Null"
                continue
            dict_outputs[doc_key] = output

        # exact match


        TP = {relation: 0 for relation in relations}
        FP = {relation: 0 for relation in relations}
        FN = {relation: 0 for relation in relations}
        relations_misclassified_as = {
            rel: {rel2: 0 for rel2 in relations} for rel in relations
        }
        true_predictions = {relation: 0 for relation in relations}
        total_samples = {relation: 0 for relation in relations}



        for doc_key in dict_outputs:
            output = dict_outputs[doc_key]
            gold_output = gold[doc_key]
            if output == gold_output:
                TP[output] += 1
                true_predictions[output] += 1
            else:
                relations_misclassified_as[gold_output][output] += 1
                FP[output] += 1
                FN[gold_output] += 1
            total_samples[gold_output] += 1

        TP_sum = sum(TP.values())
        FP_sum = sum(FP.values())
        FN_sum = sum(FN.values())

        precision = TP_sum / (TP_sum + FP_sum) if TP_sum + FP_sum > 0 else 0
        recall = TP_sum / (TP_sum + FN_sum) if TP_sum + FN_sum > 0 else 0
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        per_class_results = {
            relation: {
                "precision": TP[relation] / (TP[relation] + FP[relation]) if TP[relation] + FP[relation] > 0 else 0,
                "recall": TP[relation] / (TP[relation] + FN[relation]) if TP[relation] + FN[relation] > 0 else 0
            } for relation in relations
        }
        per_class_f1 = {
            relation:
                2 * per_class_results[relation]["precision"] * per_class_results[relation]["recall"] /\
                (per_class_results[relation]["precision"] + per_class_results[relation]["recall"])\
                    if per_class_results[relation]["precision"] + per_class_results[relation]["recall"] > 0 else 0
            for relation in relations}
        per_class_acc = {relation: true_predictions[relation] / total_samples[relation] if total_samples[relation] > 0 else 0
                         for relation in relations}
        acc = sum([true_predictions[relation] for relation in relations]) / sum([total_samples[relation] for relation in relations])
        print("Exact match")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1: {f1}")
        print(f"Micro Accuracy{acc}")
        print("Per class results")

        # calcualte and print the classification report
        y_true = [gold[doc_key] for doc_key in dict_outputs]
        y_pred = [dict_outputs[doc_key] for doc_key in dict_outputs]
        # transfrom relations to 0, 1, 2, 3, 4, 5, 6, ...
        y_true = [relations.index(relation) for relation in y_true]
        y_pred = [relations.index(relation) for relation in y_pred]
        c = classification_report(y_true, y_pred, target_names=relations[:-1], digits=4)
        # replace any whitespace longer than 1 with a single \t
        # remove all the white space after a newline character
        c = re.sub(r"\n\s*", "\n", c)
        c = re.sub(r"\s{2,}", "\t", c)
        # replace . with ,
        c = c.replace(".", ",")
        print(c)

        # print confusion matrix formatted as a tab separated pandas dataframe
        print("Confusion matrix as a tsv file")
        print(pd.DataFrame(confusion_matrix(y_true, y_pred), index=relations[:-1], columns=relations[:-1]).to_csv(sep="\t"))

        for relation in relations:
            print(f"\tRelation: {relation}")
            print(f"\t\tPrecision: {per_class_results[relation]['precision']}")
            print(f"\t\tRecall: {per_class_results[relation]['recall']}")
            print(f"\t\tF1: {per_class_f1[relation]}")
            print(f"\t\tAccuracy: {per_class_acc[relation]}")

        # print support set
        print("Support set")
        counts = {relation: 0 for relation in relations}
        gold_counts = {relation: 0 for relation in relations}
        for doc_key in dict_outputs:
            counts[dict_outputs[doc_key]] += 1
            gold_counts[gold[doc_key]] += 1
        for relation in relations:
            print(f"\tRelation: {relation}")
            print(f"\t\tCount: {counts[relation]}")
            print(f"\t\tGold Count: {gold_counts[relation]}")

        # macro and weighted F1
        macro_f1 = sum(per_class_f1.values()) / len(per_class_f1)
        weighted_f1 = sum([per_class_f1[relation] * gold_counts[relation] for relation in relations]) / sum(gold_counts.values())
        macro_acc = np.mean(list(per_class_acc.values()))
        weighted_acc = sum([per_class_acc[relation] * gold_counts[relation] for relation in relations]) / sum(gold_counts.values())
        print(f"Macro F1: {macro_f1}")
        print(f"Weighted F1: {weighted_f1}")
        print(f"Macro Accuracy: {np.mean(list(per_class_acc.values()))}")
        print(f"Weighted Accuracy: {weighted_acc}")



        print("Relations misclassified as: (Rows are the true relations, columns are the predicted relations.)")
        # make into a pandas dataframe with the rows (index) as the true relations and the columns as the predicted relations
        relations_misclassified_as_df = pd.DataFrame(relations_misclassified_as).T
        # append gold counts the last column
        relations_misclassified_as_df["Gold"] = [counts[relation] for relation in relations]
        print(relations_misclassified_as_df)


        # save the results into a text file
        with open(f"{output_dir}{prefix}mode_{mode}_results.txt", "w") as f:
            f.write("Exact match\n")
            f.write(f"Precision: {precision}\n")
            f.write(f"Recall: {recall}\n")
            f.write(f"Micro F1: {f1}\n")
            f.write(f"Macro F1: {macro_f1}\n")
            f.write(f"Weighted F1: {weighted_f1}\n")
            f.write(f"Micro Accuracy: {acc}\n")
            f.write(f"Macro Accuracy: {macro_acc}\n")
            f.write(f"Weighted Accuracy: {weighted_acc}\n")
            relations_misclassified_as_df.to_csv(f"{output_dir}{prefix}mode_{mode}_full_miss.csv")
            f.write(f"Misclassified saved to: {output_dir}{prefix}mode_{mode}_full_miss.csv\n")
            # save to latex as well
            relations_misclassified_as_df.to_latex(f"{output_dir}{prefix}mode_{mode}_full_miss.tex")
            f.write("Per class results\n")
            for relation in relations:
                f.write(f"\tRelation: {relation}\n")
                f.write(f"\t\tPrecision: {per_class_results[relation]['precision']}\n")
                f.write(f"\t\tRecall: {per_class_results[relation]['recall']}\n")
                f.write(f"\t\tF1: {per_class_f1[relation]}\n")
                f.write(f"\t\tAccuracy: {per_class_acc[relation]}\n")

