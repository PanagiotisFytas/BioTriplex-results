# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class biotriplex_dataset:
    dataset: str = "biotriplex_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "../data/"
    use_entity_tokens_as_targets: bool = False
    entity_special_tokens: bool = False
    upweight_minority_class: bool = False
    bidirectional_attention_in_entity_tokens: bool = False
    shift_entity_tokens: bool = False
    num_of_shots: int = 4

@dataclass
class biotriplex_ner_dataset:
    dataset: str = "biotriplex_ner_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "../data/"
    use_entity_tokens_as_targets: bool = False
    entity_special_tokens: bool = False
    upweight_minority_class: bool = False
    bidirectional_attention_in_entity_tokens: bool = False
    shift_entity_tokens: bool = False
    num_of_shots: int = 4

@dataclass
class biotriplex_nerlong_dataset:
    dataset: str = "biotriplex_nerlong_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "../data/"
    use_entity_tokens_as_targets: bool = False
    entity_special_tokens: bool = False
    upweight_minority_class: bool = False
    bidirectional_attention_in_entity_tokens: bool = False
    shift_entity_tokens: bool = False
    num_of_shots: int = 4

@dataclass
class biotriplex_rel_dataset:
    dataset: str = "biotriplex_rel_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "../data/"
    use_entity_tokens_as_targets: bool = False
    entity_special_tokens: bool = False
    upweight_minority_class: bool = False
    bidirectional_attention_in_entity_tokens: bool = False
    shift_entity_tokens: bool = False
    num_of_shots: int = 4

@dataclass
class biored_dataset:
    dataset: str = "biored_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "../data/biored/"
    use_entity_tokens_as_targets: bool = False
    entity_special_tokens: bool = False
    upweight_minority_class: bool = False
    bidirectional_attention_in_entity_tokens: bool = False
    shift_entity_tokens: bool = False

@dataclass
class biored_qa_dataset:
    dataset: str = "biored_qa_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "../data/biored/"
    use_entity_tokens_as_targets: bool = False
    entity_special_tokens: bool = False
    upweight_minority_class: bool = False
    bidirectional_attention_in_entity_tokens: bool = False
    shift_entity_tokens: bool = False
    num_of_shots: int = 4

class biored_qakshot_dataset(biored_qa_dataset):
    pass

@dataclass()
class biotriplex_qa_dataset:
    dataset: str = "biotriplex_qa_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "../data/"
    use_entity_tokens_as_targets: bool = False
    entity_special_tokens: bool = False
    upweight_minority_class: bool = False
    bidirectional_attention_in_entity_tokens: bool = False
    shift_entity_tokens: bool = False
    return_neg_relations: bool = False
    general_relations: bool = False
    num_of_shots: int = 4
    group_relations = False

class biotriplex_qakshot_dataset(biotriplex_qa_dataset):
    pass