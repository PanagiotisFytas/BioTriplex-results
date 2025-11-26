from anthropic import Anthropic
import json
import time
import argparse
import os
import logging
import hashlib
from dotenv import load_dotenv
import hashlib

load_dotenv()

anthropic_models = [
    "claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20240620", "claude-3-opus-20240229",
    "claude-3-sonnet-20240229", "claude-3-haiku-20240307"
]

class ClaudeBatchProcessor:
    # sleep for 1 minute
    SLEEP_TIME = 60
    VALIDATION_SLEEP_TIME = 10

    def __init__(self, dataset, dataset_name, model, output_file, max_tokens, input_file,
                 input_file_for_failed_docs=None, temperature=0.0):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.model = model
        self.output_file = output_file
        self.input_file = input_file
        self.input_file_for_failed_docs = input_file_for_failed_docs
        # for logfile remove whatever is the suffix of the output file (if it exists) and add .log
        assert input_file.endswith(".jsonl")
        assert output_file.endswith(".jsonl")
        assert input_file_for_failed_docs.endswith(".jsonl")
        self.log_file = output_file.split(".jsonl")[0] + ".log"
        logging.basicConfig(filename=self.log_file, level=logging.INFO)
        self.max_tokens = max_tokens
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = Anthropic(api_key=self.api_key)
        self.temperature = temperature
        # Dictionary to map abbreviated IDs to original doc keys
        self.id_mapping = {}

    def get_abbreviated_id(self, doc_key):
        """
        Convert a document key to an abbreviated form that fits Claude's 64-character limit.
        Store the mapping for later reference.
        """
        # check if key is safe, i.e. it matches r'[^a-zA-Z0-9_-]'
        safe_key = False
        for char in doc_key:
            if not char.isalnum():
                safe_key = False
                break
        if len(doc_key) <= 64 and safe_key:
            return doc_key

        # Create a hash of the doc_key
        hash_obj = hashlib.md5(doc_key.encode())
        abbreviated_id = hash_obj.hexdigest()

        # Store the mapping
        self.id_mapping[abbreviated_id] = doc_key

        return abbreviated_id

    def get_original_doc_key(self, abbreviated_id):
        """
        Retrieve the original document key from an abbreviated ID.
        """
        return self.id_mapping.get(abbreviated_id, abbreviated_id)

    def prepare_batch(self, failed_docs=None):
        prompts, system_prompt = self.dataset.openai_get_all_input_prompts(skip_system=True)  # Assuming this method exists in the dataset class
        requests = []

        # Save the ID mapping to a file for persistence
        id_mapping_file = self.input_file.replace('.jsonl', '_id_mapping.json')

        for doc_key, messages in prompts.items():
            if failed_docs is not None and doc_key not in failed_docs:
                continue

            # Get an abbreviated ID that fits Claude's 64-character limit
            abbreviated_id = self.get_abbreviated_id(doc_key)

            # Create the request for Claude's batch API
            requests.append({
                "custom_id": abbreviated_id,
                "original_doc_key": doc_key,  # Keep the original for our records
                "params": {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    # "top_p": self.top_p,
                    "system": system_prompt,
                }
            })

        # Save the requests to file for reference
        target_file = self.input_file if failed_docs is None else self.input_file_for_failed_docs
        with open(target_file, "w") as f:
            for request in requests:
                f.write(json.dumps(request) + "\n")

        # Save the ID mapping for later reference
        with open(id_mapping_file, "w") as f:
            json.dump(self.id_mapping, f)

        return requests

    def load_id_mapping(self):
        """Load the ID mapping from file if it exists"""
        id_mapping_file = self.input_file.replace('.jsonl', '_id_mapping.json')
        if os.path.exists(id_mapping_file):
            with open(id_mapping_file, "r") as f:
                self.id_mapping = json.load(f)

    def submit_batch(self, failed_docs=None):
        # Load the ID mapping in case we're resuming a previous operation
        self.load_id_mapping()

        requests = self.prepare_batch(failed_docs=failed_docs)

        # Convert the requests to the format expected by the Claude API
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        api_requests = [
            Request(
                custom_id=req["custom_id"],  # Using the abbreviated ID
                params=MessageCreateParamsNonStreaming(
                    model=req["params"]["model"],
                    max_tokens=req["params"]["max_tokens"],
                    temperature=req["params"]["temperature"],
                    # top_p=req["params"]["top_p"],
                    messages=req["params"]["messages"],
                    system=req["params"]["system"],
                )
            )
            for req in requests
        ]

        # Submit the batch
        batch = self.client.messages.batches.create(requests=api_requests)

        print(f"Batch {batch.id} created with {len(requests)} requests")
        logging.info(f"Batch {batch.id} created with {len(requests)} requests")
        return batch

    def check_batch_status(self, batch_id):
        return self.client.messages.batches.retrieve(batch_id)

    def wait_for_completion(self, batch_id):
        while True:
            batch = self.check_batch_status(batch_id)
            if batch.processing_status == "ended":
                print(f"Batch {batch_id} completed")
                logging.info(f"Batch {batch_id} completed successfully")
                logging.info(f"All info: {batch}")
                break
            else:
                print(f"Batch {batch_id} status: {batch.processing_status}, "
                      f"Processing: {batch.request_counts.processing}, "
                      f"Succeeded: {batch.request_counts.succeeded}, "
                      f"Errored: {batch.request_counts.errored}")
                logging.info(f"Batch {batch_id} status: {batch.processing_status}, "
                             f"Processing: {batch.request_counts.processing}, "
                             f"Succeeded: {batch.request_counts.succeeded}, "
                             f"Errored: {batch.request_counts.errored}")
                time.sleep(self.SLEEP_TIME)
        return batch

    def cancel_batch(self, batch_id):
        return self.client.messages.batches.cancel(batch_id)

    def retrieve_results(self, batch_id):
        # Load the ID mapping to ensure we can translate back to original doc keys
        self.load_id_mapping()

        batch = self.check_batch_status(batch_id)
        if batch.processing_status != "ended":
            print(f"Batch {batch_id} is not complete yet. Current status: {batch.processing_status}")
            return None

        # Stream the results as per Claude's recommendation
        results = list(self.client.messages.batches.results(batch_id))
        return results

    def save_results(self, batch_id):
        results = self.retrieve_results(batch_id)
        if not results:
            return

        # Load the original requests to get the mapping from abbreviated_id back to original doc_key
        request_doc_keys = {}
        with open(self.input_file, "r") as f:
            for line in f:
                req = json.loads(line)
                request_doc_keys[req["custom_id"]] = req.get("original_doc_key", req["custom_id"])

        successful_docs = []
        all_docs = list(request_doc_keys.values())

        outputs = []
        for result in results:
            abbreviated_id = result.custom_id

            # Map back to the original doc_key
            doc_key = request_doc_keys.get(abbreviated_id, abbreviated_id)
            # If not found in request_doc_keys, try the id_mapping
            if doc_key == abbreviated_id and abbreviated_id in self.id_mapping:
                doc_key = self.id_mapping[abbreviated_id]

            if result.result.type == "succeeded":
                # Get the text content from the Claude message
                message_content = []
                for content_item in result.result.message.content:
                    if content_item.type == "text":
                        message_content.append(content_item.text)

                outputs.append({
                    "doc_key": doc_key,  # Using the original doc_key
                    "output": message_content
                })
                successful_docs.append(doc_key)
            else:
                error_type = result.result.type
                error_message = "Unknown error"
                if error_type == "errored" and hasattr(result.result, "error"):
                    error_message = result.result.error.message if hasattr(result.result.error, "message") else str(result.result.error)

                logging.error(f"Failed to process doc {doc_key}: {error_type} - {error_message}")

        # Save as line separated json
        with open(self.output_file, "w") as f:
            for output in outputs:
                f.write(json.dumps(output) + "\n")

        failed_docs = list(set(all_docs) - set(successful_docs))
        # Print percentage of successful docs
        print(f"Percentage of successful docs: {len(successful_docs) / len(all_docs)}")
        logging.info(f"Percentage of successful docs: {len(successful_docs) / len(all_docs)}")
        # Print location of output file
        print(f"Output file saved at {self.output_file}")
        logging.info(f"Output file saved at {self.output_file}")
        return failed_docs

    def return_failed_docs(self):
        # Load the ID mapping to ensure we have the original doc keys
        self.load_id_mapping()

        # Get all original doc keys from the input file
        all_docs = []
        with open(self.input_file, "r") as f:
            for line in f:
                req = json.loads(line)
                all_docs.append(req.get("original_doc_key", req["custom_id"]))

        # Get the successful doc keys from the output file
        successful_docs = []
        if os.path.exists(self.output_file):
            with open(self.output_file, "r") as f:
                for line in f:
                    successful_docs.append(json.loads(line)["doc_key"])

        failed_docs = list(set(all_docs) - set(successful_docs))
        return failed_docs

    def save_results_after_failed_docs(self, batch_id):
        results = self.retrieve_results(batch_id)
        if not results:
            return

        # Load the original requests for failed docs to get the mapping
        request_doc_keys = {}
        with open(self.input_file_for_failed_docs, "r") as f:
            for line in f:
                req = json.loads(line)
                request_doc_keys[req["custom_id"]] = req.get("original_doc_key", req["custom_id"])

        previously_successful_docs = []
        if os.path.exists(self.output_file):
            with open(self.output_file, "r") as f:
                for line in f:
                    line = json.loads(line)
                    previously_successful_docs.append(line["doc_key"])

        all_docs = []
        with open(self.input_file, "r") as f:
            for line in f:
                req = json.loads(line)
                all_docs.append(req.get("original_doc_key", req["custom_id"]))

        new_successful_docs = []
        outputs = []

        for result in results:
            abbreviated_id = result.custom_id

            # Map back to the original doc_key
            doc_key = request_doc_keys.get(abbreviated_id, abbreviated_id)
            # If not found in request_doc_keys, try the id_mapping
            if doc_key == abbreviated_id and abbreviated_id in self.id_mapping:
                doc_key = self.id_mapping[abbreviated_id]

            if result.result.type == "succeeded":
                # Get the text content from the Claude message
                message_content = []
                for content_item in result.result.message.content:
                    if content_item.type == "text":
                        message_content.append(content_item.text)

                outputs.append({
                    "doc_key": doc_key,  # Using the original doc_key
                    "output": message_content
                })
                new_successful_docs.append(doc_key)
            else:
                error_type = result.result.type
                error_message = "Unknown error"
                if error_type == "errored" and hasattr(result.result, "error"):
                    error_message = result.result.error.message if hasattr(result.result.error, "message") else str(result.result.error)

                logging.error(f"Failed to process doc {doc_key}: {error_type} - {error_message}")

        # Save as line separated json, appending to existing file
        with open(self.output_file, "a") as f:
            for output in outputs:
                f.write(json.dumps(output) + "\n")

        # Print percentage of successful docs
        print(f"Percentage of successful docs: {(len(previously_successful_docs) + len(new_successful_docs)) / len(all_docs)}")
        logging.info(f"Percentage of successful docs: {(len(previously_successful_docs) + len(new_successful_docs)) / len(all_docs)}")
        # Print location of output file
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
    parser.add_argument("--model", type=str, choices=anthropic_models, default="claude-3-7-sonnet-20250219")
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
    parser.add_argument("--general_relations", action="store_true", default=False)
    # parser.add_argument("--top_p", type=float, default=0.8)
    args = parser.parse_args()

    genrel = "" if not args.general_relations else f"_general_relations_"
    file_prefix = (f"{args.dataset}_{args.dataset_mode}_{args.model}_{args.num_of_shots}shots_{args.max_tokens}tokens_"
                   f"{args.temperature}temp_{genrel}")
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

    batch_processor = ClaudeBatchProcessor(dataset, args.dataset, args.model, output_file, args.max_tokens, input_file,
                                           input_file_for_failed_docs, temperature=args.temperature)

    if args.debug:
        # Just prepare the batch
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
        print(f"Batch {args.batch_id} status: {batch.processing_status}")
        print(f"Request counts: {batch.request_counts}")
        logging.info(f"Batch {args.batch_id} status: {batch.processing_status}")
        logging.info(f"Request counts: {batch.request_counts}")
    else:
        raise Exception("Invalid mode")


if __name__ == "__main__":
    main()

"""
python src/api/datasets/api_model_scripts/claude_submit.py --num_of_shots=5 --group_relations --mode=submit_and_wait
python src/api/datasets/api_model_scripts/claude_submit.py --num_of_shots=5 --mode=submit_and_wait --dataset=biored_qakshot --max_tokens=5
python src/api/datasets/api_model_scripts/claude_submit.py --num_of_shots=0 --mode=submit_and_wait --dataset=biored_qakshot --max_tokens=5
python src/api/datasets/api_model_scripts/claude_submit.py --num_of_shots=0 --group_relations --mode=submit_and_wait --general_relations
python src/api/datasets/api_model_scripts/claude_submit.py --num_of_shots=5 --group_relations --mode=submit_and_wait --general_relations

"""