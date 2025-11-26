from openai import OpenAI
import json
import time
import argparse
import os
import logging
from dotenv import load_dotenv

load_dotenv()


openai_models = [
    "gpt-4o", "gpt-4o-2024-08-06", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-4-32k", "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k", "gpt-4-turbo-preview", "gpt-4-vision-preview", "gpt-4-turbo-2024-04-09", "gpt-4-0314",
    "gpt-4-32k-0314", "gpt-4-32k-0613", "gpt-3.5-turbo-0301", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0613"
]

class OpenAIBatchProcessor:
    # sleep for 1 minute
    SLEEP_TIME = 60
    VALIDATION_SLEEP_TIME = 10
    def __init__(self, dataset, dataset_name, model, output_file, max_tokens, input_file,
                 input_file_for_failed_docs=None, temperature=0.0, top_p=1.0):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.model = model
        self.output_file = output_file
        self.input_file = input_file
        self.input_file_for_failed_docs = input_file_for_failed_docs
        # for logfile remove whaterver is the suffix of the output file (if it exists) and add .log
        assert input_file.endswith(".jsonl")
        assert output_file.endswith(".jsonl")
        assert input_file_for_failed_docs.endswith(".jsonl")
        self.log_file = output_file.split(".jsonl")[0] + ".log"
        logging.basicConfig(filename=self.log_file, level=logging.INFO)
        self.max_tokens = max_tokens
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api = OpenAI(api_key=self.api_key)
        self.temperature = temperature
        self.top_p = top_p

    def prepare_batch(self, failed_docs=None):
        prompts = self.dataset.openai_get_all_input_prompts()
        input_data = []

        for doc_key, messages in prompts.items():
            if failed_docs is not None and doc_key not in failed_docs:
                continue
            input_data.append({
                "custom_id": doc_key,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                },
            })

        # Batches start with a .jsonl file where each line contains the details of an individual request to the API. For now, the available endpoints are /v1/chat/completions (Chat Completions API) and /v1/embeddings (Embeddings API). For a given input file, the parameters in each line's body field are the same as the parameters for the underlying endpoint. Each request must include a unique custom_id value, which you can use to reference results after completion. Here's an example of an input file with 2 requests. Note that each input file can only include requests to a single model.
        target_file = self.input_file if failed_docs is None else self.input_file_for_failed_docs
        with open(target_file, "w") as f:
            for line in input_data:
                f.write(json.dumps(line) + "\n")

    def upload_batch(self, failed_docs=None):
        # assert input file exists
        if failed_docs is None:
            assert os.path.exists(self.input_file)
        else:
            assert os.path.exists(self.input_file_for_failed_docs)
        batch_input_file = self.api.files.create(
            file=open(self.input_file, "rb") if failed_docs is None else open(self.input_file_for_failed_docs, "rb"),
            purpose="batch"
        )
        return batch_input_file

    def create_batch(self, batch_input_file):
        # create a batch
        batch_input_file_id = batch_input_file.id
        return self.api.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "model": self.model,
                "max_tokens": str(self.max_tokens),
                "dataset_name": self.dataset_name,
                "output_file": self.output_file
            }
        )

    def submit_batch(self, failed_docs=None):
        self.prepare_batch(failed_docs=failed_docs)
        batch_input_file = self.upload_batch(failed_docs=failed_docs)
        print(f"Batch input file {batch_input_file.id} uploaded")
        logging.info(f"Batch input file {batch_input_file.id} uploaded")
        batch = self.create_batch(batch_input_file)
        print(f"Batch {batch.id} created")
        logging.info(f"Batch {batch.id} created")
        return batch

    def check_batch_status(self, batch_id):
        return self.api.batches.retrieve(batch_id)

    def wait_for_completion(self, batch_id):
        while True:
            batch = self.check_batch_status(batch_id)
            if batch.status == "completed":
                print(f"Batch {batch_id} completed")
                logging.info(f"Batch {batch_id} completed successfully")
                logging.info(f"All info: {batch}")
                break
            elif batch.status in ["failed", "expired", "cancelled"]:
                print(f"Batch {batch_id} failed with status {batch.status}")
                logging.error(f"Batch {batch_id} failed with status {batch.status}")
                logging.error(f"All info: {batch}")
                raise Exception(f"Batch {batch_id} failed with status {batch.status}")
            elif batch.status == "validating":
                time.sleep(self.VALIDATION_SLEEP_TIME)
            else:
                time.sleep(self.SLEEP_TIME)
        return batch

    def cancel_batch(self, batch_id):
        return self.api.batches.cancel(batch_id)

    def retrieve_results(self, batch_id):
        output_file_id = self.check_batch_status(batch_id).output_file_id
        assert output_file_id is not None
        file_response = self.api.files.content(output_file_id)
        return file_response

    def save_results(self, batch_id):
        results = self.retrieve_results(batch_id)
        successful_docs = []
        all_docs = []
        with open(self.input_file, "r") as f:
            for line in f:
                all_docs.append(json.loads(line)["custom_id"])
        outputs = []
        for line in results.iter_lines():
            line = json.loads(line)
            doc_key = line["custom_id"]
            output = [line["response"]["body"]["choices"][id]["message"]["content"]
                        for id in range(len(line["response"]["body"]["choices"]))]  # get all completions
            outputs.append({
                "doc_key": doc_key,
                "output": output
            })
            successful_docs.append(doc_key)
        # save as line separated json
        with open(self.output_file, "w") as f:
            for output in outputs:
                f.write(json.dumps(output) + "\n")
        failed_docs = list(set(all_docs) - set(successful_docs))
        # print percentage of successful docs
        print(f"Percentage of successful docs: {len(successful_docs) / len(all_docs)}")
        logging.info(f"Percentage of successful docs: {len(successful_docs) / len(all_docs)}")
        # print location of output file
        print(f"Output file saved at {self.output_file}")
        logging.info(f"Output file saved at {self.output_file}")

    def return_failed_docs(self):
        all_docs = []
        with open(self.input_file, "r") as f:
            for line in f:
                all_docs.append(json.loads(line)["custom_id"])
        successful_docs = []
        with open(self.output_file, "r") as f:
            for line in f:
                successful_docs.append(json.loads(line)["doc_key"])
        failed_docs = list(set(all_docs) - set(successful_docs))
        return failed_docs

    def save_results_after_failed_docs(self, batch_id):
        results = self.retrieve_results(batch_id)
        previously_successful_docs = []
        with open(self.output_file, "r") as f:
            for line in f:
                line = json.loads(line)
                previously_successful_docs.append(line["doc_key"])
        all_docs = []
        with open(self.input_file, "r") as f:
            for line in f:
                all_docs.append(json.loads(line)["custom_id"])
        new_successful_docs = []
        outputs = []
        failed_docs = []
        with open(self.input_file_for_failed_docs, "r") as f:
            for line in f:
                failed_docs.append(json.loads(line)["custom_id"])
        for line in results.iter_lines():
            line = json.loads(line)
            doc_key = line["custom_id"]
            if doc_key in failed_docs:
                output = [line["response"]["body"]["choices"][id]["message"]["content"]
                          for id in range(len(line["response"]["body"]["choices"]))]  # get all completions
                outputs.append({
                    "doc_key": doc_key,
                    "output": output
                })
                new_successful_docs.append(doc_key)
            else:
                raise Exception(f"Doc key {doc_key} not in failed docs")
        # save as line separated json
        with open(self.output_file, "a") as f:
            for output in outputs:
                f.write(json.dumps(output) + "\n")
        # print percentage of successful docs
        print(f"Percentage of successful docs: {(len(previously_successful_docs) + len(new_successful_docs)) / len(all_docs)}")
        logging.info(f"Percentage of successful docs: {(len(previously_successful_docs) + len(new_successful_docs)) / len(all_docs)}")
        # print location of output file
        print(f"Output file saved at {self.output_file}")
        logging.info(f"Output file saved at {self.output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        choices=["biotriplex_qaksot", "biotriplex_qa0sot", "biored_qakshot"],
                        default="biotriplex_qaksot")
    parser.add_argument("--num_of_shots", type=int, default=4)
    parser.add_argument("--group_relations", action="store_true", default=False)
    parser.add_argument("--dataset_mode", type=str, default="test", choices=["test", "train", "val"])
    parser.add_argument("--model", type=str, choices=openai_models, default="gpt-4")
    parser.add_argument("--output_file_suffix", type=str,
                        default="_output.jsonl")
    parser.add_argument("--max_tokens", type=int, default=50)
    parser.add_argument("--input_file_suffix", type=str, default="_input.jsonl")
    parser.add_argument("--input_file_for_failed_docs_suffix", type=str, default="_input_failed_docs.jsonl")
    parser.add_argument("--mode", type=str, required=False, default="submit",
                        choices=["submit", "submit_and_wait", "cancel", "retrieve", "check_status"])
    parser.add_argument("--batch_id", type=str, default=None)
    parser.add_argument("--failed_docs", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--general_relations", action="store_true", default=False)
    args = parser.parse_args()
    genrel = "" if not args.general_relations else f"_general_relations_"
    file_prefix = (f"{args.dataset}_{args.dataset_mode}_{args.model}_{args.num_of_shots}shots_{args.max_tokens}tokens_"
                   f"{args.temperature}temp_{args.top_p}topp{genrel}")
    if args.group_relations:
        file_prefix += "_group_relations"
    output_file = f"{file_prefix}{args.output_file_suffix}"
    input_file = f"{file_prefix}{args.input_file_suffix}"
    input_file_for_failed_docs = f"{file_prefix}{args.input_file_for_failed_docs_suffix}"
    if args.dataset == "biotriplex_qaksot":
        from api.datasets.biotriplex_qakshot_dataset import BioTriplexQADataset
        from api.datasets.configs.datasets import biotriplex_qa_dataset
        biotriplex_qa_dataset.num_of_shots = args.num_of_shots
        biotriplex_qa_dataset.group_relations = args.group_relations
        biotriplex_qa_dataset.general_relations = args.general_relations
        dataset = BioTriplexQADataset(biotriplex_qa_dataset, None, args.dataset_mode, max_words=None)
    elif args.dataset == "biored_qakshot":
        from api.datasets.biored_qakshot_dataset import BioRedQADataset
        from api.datasets.configs.datasets import biored_qakshot_dataset
        biored_qakshot_dataset.num_of_shots = args.num_of_shots
        dataset = BioRedQADataset(biored_qakshot_dataset, None, args.dataset_mode, max_words=None)
    else:
        raise Exception("Invalid dataset")
    batch_processor = OpenAIBatchProcessor(dataset, args.dataset, args.model, output_file, args.max_tokens, input_file,
                                           input_file_for_failed_docs, temperature=args.temperature, top_p=args.top_p)
    if args.debug:
        # just prepare the batch
        batch_processor.prepare_batch()
    elif args.mode == "submit":
        if args.failed_docs:
            failed_docs = batch_processor.return_failed_docs()
            batch = batch_processor.submit_batch(failed_docs=failed_docs)
        else:
            batch = batch_processor.submit_batch()
        print(f"Batch id: {batch.id}")
        logging.info(f"Batch id: {batch.id}")
    elif args.mode == "submit_and_wait":
        if args.failed_docs:
            failed_docs = batch_processor.return_failed_docs()
            batch = batch_processor.submit_batch(failed_docs=failed_docs)
            batch_id = batch.id
            batch_processor.wait_for_completion(batch_id)
            batch_processor.save_results_after_failed_docs(batch_id)
        else:
            batch = batch_processor.submit_batch()
            batch_id = batch.id
            batch_processor.wait_for_completion(batch_id)
            batch_processor.save_results(batch_id)
    elif args.mode == "cancel":
        assert args.batch_id is not None
        batch_processor.cancel_batch(args.batch_id)
    elif args.mode == "retrieve":
        assert args.batch_id is not None
        if args.failed_docs:
            batch_processor.save_results_after_failed_docs(args.batch_id)
        else:
            batch_processor.save_results(args.batch_id)
    elif args.mode == "check_status":
        assert args.batch_id is not None
        batch = batch_processor.check_batch_status(args.batch_id)
        print(f"Batch {args.batch_id} status: {batch.status}")
        logging.info(f"Batch {args.batch_id} status: {batch.status}")
    else:
        raise Exception("Invalid mode")


if __name__ == "__main__":
    main()

    """
    Batch id: batch_679ba97da988819090908ca2496f6389  --> 0 shot done
    Batch id: batch_679babb69cdc81909ac01ead10f000b9  --> 1 shot done
    Batch id: batch_679badec391c8190abf5fa55e4ca3cb6  --> 2 shot done
    Batch id: batch_679bae122938819099876be25db71f68  --> 3 shot done
    Batch id: batch_679bae297c9481909dde17385961c85e  --> 4 shot done 
    Batch id: batch_679cc9c1adb08190b1d240183276f011  --> 5 shot-group cancelled
    Batch id: batch_679cd57c4f6c819080619eba29cb7a5d --> 0 shot-group cancelled
    """