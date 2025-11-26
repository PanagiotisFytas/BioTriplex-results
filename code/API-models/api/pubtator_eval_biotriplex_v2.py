#!/usr/bin/env python3
"""
Evaluate PubTator gene/disease NER against your XML annotations.

Ground truth:
    ../data/Annotated Full Text Paper Folders/<pmcid>/*.xml

Test set:
    ../data/test_shorter.txt   (JSONL, with "doc_key" like "8508478_5. Conclusions")

PubTator:
    https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/pmc_export/biocxml
"""

import json
import requests
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List
from collections import Counter


PUBTATOR_URL = (
    "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/"
    "publications/pmc_export/biocxml"
)


@dataclass(frozen=True)
class Entity:
    start: int
    end: int
    label: str  # "GENE" or "DISEASE"
    text: str


def normalize_label(label: str) -> str:
    lab = label.strip().upper()
    if lab.startswith("GENE"):
        return "GENE"
    if lab.startswith("DISEASE"):
        return "DISEASE"
    return lab

def normalize_text(text: str) -> str:
    """
    Normalize mention strings for comparison:

    - lowercase
    - strip leading/trailing whitespace
    - collapse internal whitespace to single spaces
    """
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text




# ---------------- Test set handling -----------------


def load_test_pmcids(test_json_path: Path) -> Set[str]:
    """Read JSONL test file and return unique PMC IDs as strings (no 'PMC' prefix)."""
    pmcids: Set[str] = set()
    with test_json_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            # Only keep explicit test split if present
            split = ex.get("_split", [])
            if split and "test" not in split:
                continue

            doc_key = ex["doc_key"]
            # Expect format like "8508478_5. Conclusions"
            num_part = doc_key.split("_", 1)[0]
            if not num_part.isdigit():
                raise ValueError(f"Unexpected doc_key format: {doc_key}")
            pmcids.add(num_part)

    if not pmcids:
        raise RuntimeError("No test PMCIDs found in test file.")

    return pmcids


# ------------- Your XML parsing ---------------------


def find_my_xml_for_pmc(pmcid: str, xml_root: Path) -> List[Path]:
    """
    Return all XML files for this PMC ID.

    ### IMPORTANT ASSUMPTION
    Right now this just returns every '*.xml' in the folder
    '../data/Annotated Full Text Paper Folders/<pmcid>/'.
    If your structure is different (e.g. subfolders/section files),
    adjust this.
    """
    pmc_dir = xml_root / pmcid
    if not pmc_dir.is_dir():
        raise FileNotFoundError(f"Ground-truth directory not found for PMC {pmcid}: {pmc_dir}")

    xml_files = sorted(pmc_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML files found for PMC {pmcid} in {pmc_dir}")

    return xml_files


def parse_my_single_xml(xml_path: Path) -> List[Entity]:
    """Parse one of your Genomics_ConceptTask XML files into Entity objects.

    Handles multi-span annotations like '1102~1109,1138~1144' or
    '1102~1109;1138~1144' by turning each fragment into a separate Entity.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    tags = root.find("TAGS")
    if tags is None:
        raise ValueError(f"No <TAGS> element in {xml_path}")

    entities: List[Entity] = []

    for elem in tags:
        if elem.tag not in ("GENE", "DISEASE"):
            continue

        label = elem.tag  # already uppercase in your format
        spans_attr = elem.get("spans")
        text_attr = elem.get("text", "")

        if not spans_attr:
            raise ValueError(f"Missing 'spans' attribute in {elem.tag} in {xml_path}")

        # Allow commas or semicolons as separators for non-contiguous spans
        # e.g. "1102~1109,1138~1144" or "1102~1109;1138~1144"
        span_parts = re.split(r"[;,]", spans_attr)

        for span_part in span_parts:
            span_part = span_part.strip()
            if not span_part:
                continue
            try:
                start_str, end_str = span_part.split("~")
                start, end = int(start_str), int(end_str)
            except Exception as e:
                raise ValueError(
                    f"Bad spans format '{spans_attr}' in {xml_path}"
                ) from e

            entities.append(
                Entity(start=start, end=end, label=label, text=text_attr)
            )

    return entities

def load_all_my_annotations(
    pmcids: Set[str], xml_root: Path
) -> Dict[str, List[Entity]]:
    """Load and merge all your XML annotations, grouped by pmcid."""
    gt: Dict[str, List[Entity]] = {}

    for pmcid in pmcids:
        xml_files = find_my_xml_for_pmc(pmcid, xml_root)
        doc_entities: List[Entity] = []
        for xml_path in xml_files:
            doc_entities.extend(parse_my_single_xml(xml_path))

        if not doc_entities:
            raise RuntimeError(f"No entities found in XML for PMC {pmcid}")

        gt[pmcid] = doc_entities

    return gt


# ------------- PubTator BioC parsing ----------------


def fetch_pubtator_chunk(pmcids_chunk: List[str]) -> str:
    """
    Fetch BioC XML for a chunk of PMC IDs (without 'PMC' prefix).

    Crashes if HTTP status is not ok.
    """
    if not pmcids_chunk:
        return ""

    pmcids_param = ",".join(f"PMC{pid}" if not pid.startswith("PMC") else pid
                            for pid in pmcids_chunk)

    resp = requests.get(PUBTATOR_URL, params={"pmcids": pmcids_param})
    resp.raise_for_status()
    return resp.text


def parse_pubtator_biocxml(xml_str: str) -> Dict[str, List[Entity]]:
    """
    Parse PubTator BioC XML string into entities, grouped by pmcid (no 'PMC').

    Only keeps Gene / Disease.
    """
    root = ET.fromstring(xml_str)
    ns = ""  # BioC has no namespaces normally

    by_pmc: Dict[str, List[Entity]] = {}

    for doc in root.findall("./document"):
        doc_id = doc.findtext("id")
        if not doc_id:
            raise ValueError("PubTator document without <id>")

        pmcid = doc_id.replace("PMC", "")
        if not pmcid.isdigit():
            raise ValueError(f"Unexpected PubTator document id: {doc_id}")

        entities: List[Entity] = []

        # In BioC, annotations are nested inside passages
        for passage in doc.findall("passage"):
            passage_offset_txt = passage.findtext("offset")
            if passage_offset_txt is None:
                raise ValueError(f"Missing passage offset in PubTator for {doc_id}")
            passage_offset = int(passage_offset_txt)

            for ann in passage.findall("annotation"):
                # Find type infon
                ann_type = None
                for infon in ann.findall("infon"):
                    if infon.get("key") == "type":
                        ann_type = infon.text
                        break
                if ann_type is None:
                    continue

                label = normalize_label(ann_type)
                if label not in ("GENE", "DISEASE"):
                    continue

                ann_text = ann.findtext("text") or ""

                loc = ann.find("location")
                if loc is None:
                    raise ValueError(f"Annotation without <location> in {doc_id}")
                loc_offset = int(loc.attrib["offset"])
                loc_length = int(loc.attrib["length"])

                start = passage_offset + loc_offset
                end = start + loc_length

                entities.append(
                    Entity(start=start, end=end, label=label, text=ann_text)
                )

        if not entities:
            # Let this be loud — user requested crashes for missing sections
            raise RuntimeError(f"PubTator returned no Gene/Disease entities for {doc_id}")

        by_pmc[pmcid] = entities

    return by_pmc


def fetch_pubtator_for_all(
    pmcids: Set[str], chunk_size: int = 50
) -> Dict[str, List[Entity]]:
    """Fetch PubTator annotations for all given PMC IDs."""
    pmcids_list = sorted(pmcids)
    result: Dict[str, List[Entity]] = {}

    for i in range(0, len(pmcids_list), chunk_size):
        chunk = pmcids_list[i : i + chunk_size]
        xml_str = fetch_pubtator_chunk(chunk)
        chunk_map = parse_pubtator_biocxml(xml_str)

        # Ensure all requested IDs are present
        missing = set(chunk) - set(chunk_map.keys())
        if missing:
            raise RuntimeError(
                f"PubTator response missing the following PMCIDs: {sorted(missing)}"
            )

        result.update(chunk_map)

    return result


# ---------------- Evaluation ------------------------


def compute_prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1



def evaluate_ner(
    gt_by_doc: Dict[str, List[Entity]],
    pred_by_doc: Dict[str, List[Entity]],
    pmcids: Set[str],
) -> None:
    """
    Compute micro P/R/F1 for GENE and DISEASE using *text-only* matching.

    For each PMC + label:
      - build a Counter over normalized mention texts (bags of strings)
      - TP = sum of per-string min(gt_count, pred_count)
      - FP/FN computed from the differences in counts

    We still crash loudly if any PMC is missing on either side.
    """

    # Sanity checks – crash loudly if anything is missing
    missing_gt = pmcids - set(gt_by_doc.keys())
    if missing_gt:
        raise RuntimeError(f"Missing ground truth for pmcids: {sorted(missing_gt)}")

    missing_pred = pmcids - set(pred_by_doc.keys())
    if missing_pred:
        raise RuntimeError(f"Missing PubTator predictions for pmcids: {sorted(missing_pred)}")

    for label in ("GENE", "DISEASE"):
        tp = fp = fn = 0

        for pmcid in pmcids:
            # Build bag-of-strings for GT and predictions
            gt_counter = Counter(
                normalize_text(e.text)
                for e in gt_by_doc[pmcid]
                if e.label == label
            )
            pred_counter = Counter(
                normalize_text(e.text)
                for e in pred_by_doc[pmcid]
                if e.label == label
            )

            # Remove empty texts if any (paranoid)
            if "" in gt_counter:
                del gt_counter[""]
            if "" in pred_counter:
                del pred_counter[""]

            # Union of all strings seen in this doc for this label
            all_strings = set(gt_counter.keys()) | set(pred_counter.keys())

            for s in all_strings:
                g = gt_counter.get(s, 0)
                p = pred_counter.get(s, 0)
                tp += min(g, p)
                fp += max(p - g, 0)
                fn += max(g - p, 0)

        precision, recall, f1 = compute_prf(tp, fp, fn)
        print(
            f"{label} (string-based): "
            f"P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}  "
            f"(TP={tp} FP={fp} FN={fn})"
        )

# ---------------- Main -----------------------------


def main():
    test_json_path = Path("../data/test_shorter.txt")
    xml_root = Path("../data/Annotated Full Text Paper Folders")

    print(f"Reading test set from {test_json_path}")
    pmcids = load_test_pmcids(test_json_path)
    print(f"Found {len(pmcids)} unique test PMC IDs")

    print("Loading ground-truth XML annotations...")
    gt_by_doc = load_all_my_annotations(pmcids, xml_root)

    print("Fetching PubTator annotations...")
    pred_by_doc = fetch_pubtator_for_all(pmcids, chunk_size=50)

    print("Evaluating...")
    evaluate_ner(gt_by_doc, pred_by_doc, pmcids)


if __name__ == "__main__":
    main()
