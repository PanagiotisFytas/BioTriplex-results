# python
import argparse
import json
import os
import re
import sys
import time
import platform
import xml.etree.ElementTree as ET
from typing import Dict, List, Set, Tuple, Optional, Any

try:
    import requests
except Exception as e:
    raise SystemExit("Missing dependency 'requests'. Install with: python -m pip install requests") from e

# ------------------------- Config ------------------------- #

# PubTator3 publications (PubMed) export endpoint: title/abstract only (BioC JSON)
PUBTATOR3_PUBLICATIONS_EXPORT_BASE = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export"
# PubTator3 PMC publications full-text export endpoint (BioC XML)
PUBTATOR3_PUBLICATIONS_PMC_EXPORT_BASE = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/pmc_export"

# Concepts we care about for PubMed title/abstract
CONCEPTS = ("gene", "disease")

# politeness and timeouts
HTTP_TIMEOUT_SEC = 60
DEBUG = True  # set by CLI

# NCBI PMC idconv (PMC -> PMID)
PMC_IDCONV_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"

# --- caches ---
_PUB_DOC_CACHE_BY_PMID: Dict[str, dict] = {}
_PUB_DOC_CACHE_BY_PMCID_XMLDOCS: Dict[str, List[dict]] = {}
_PMC2PMID_CACHE: Dict[str, str] = {}


def debug_print(msg: str) -> None:
    if DEBUG:
        sys.stderr.write(str(msg).rstrip() + "\n")
        sys.stderr.flush()


# ------------------------- Section normalization ------------------------- #

_ROMAN_RE = re.compile(r"^(?=[MDCLXVI])(M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3}))$", re.IGNORECASE)

def _strip_leading_numbering(s: str) -> str:
    # Remove patterns like "1. ", "1.2.3 - ", "I. ", "II) "
    t = s.strip()
    # numeric like "1.2.3"
    t = re.sub(r"^\s*\d+(?:\.\d+)*\s*[\)\.\-:]*\s*", "", t)
    # roman numerals like "II. ", "IV) "
    parts = t.split(maxsplit=1)
    if parts:
        head = parts[0].strip(").:-")
        if _ROMAN_RE.match(head):
            t = parts[1] if len(parts) > 1 else ""
    return t.strip()

def _normalize_token_space(s: str) -> str:
    # Lowercase, replace separators with spaces, keep letters only for robust matching
    t = s.lower()
    t = t.replace("_", " ").replace("-", " ").replace("/", " ").replace("\\", " ")
    # remove brackets/punctuation to stabilize variants
    t = re.sub(r"[^a-z\s]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# map several variants to a canonical section name
_SECTION_SYNONYMS: Dict[str, str] = {
    "article title": "TITLE",
    "title": "TITLE",
    "abstract": "ABSTRACT",
    "abstract text": "ABSTRACT",
    "introduction": "INTRODUCTION",
    "background": "INTRODUCTION",
    "materials and methods": "METHODS",
    "methods and materials": "METHODS",
    "methods": "METHODS",
    "results": "RESULTS",
    "discussion": "DISCUSSION",
    "results and discussion": "RESULTS AND DISCUSSION",
    "conclusion": "CONCLUSIONS",
    "conclusions": "CONCLUSIONS",
    "supplementary": "SUPPLEMENTARY",
    "acknowledgments": "ACKNOWLEDGMENTS",
    "acknowledgements": "ACKNOWLEDGMENTS",
}

def canonicalize_section_name(raw: str) -> str:
    if not raw:
        return ""
    t = _strip_leading_numbering(str(raw))
    t = _normalize_token_space(t)
    if not t:
        return ""
    if t in _SECTION_SYNONYMS:
        return _SECTION_SYNONYMS[t]
    if t.startswith("abstract"):
        return "ABSTRACT"
    if t.startswith("article title") or t == "title":
        return "TITLE"
    return t.upper()

# infon keys to check for section labels in PubTator3 passages
_INFON_SECTION_KEYS = ("section", "section_type", "type", "passage_type", "title")

def passage_section_canonical(infons: Dict[str, Any]) -> str:
    for k in _INFON_SECTION_KEYS:
        v = infons.get(k)
        if v:
            c = canonicalize_section_name(str(v))
            if c:
                return c
    return ""


# ------------------------- CLI ------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PubTator3 vs BioTriplEx XML gene/disease annotations using PMC numeric codes and PubTator3 exports."
    )
    parser.add_argument(
        "--test-jsonl",
        help="Path to test JSONL/TXT that contains PMC ids (lines containing 'PMC<digits>' anywhere).",
        required=False,
        default="../data/test_shorter.txt",
    )
    parser.add_argument(
        "--xml-root",
        help="Root directory of 'Annotated Full Text Paper Folders' (contains PMC* dirs or numeric dirs).",
        required=False,
        default="../data/Annotated Full Text Paper Folders",
    )
    parser.add_argument(
        "--output-gold",
        help="Output JSONL file with gold and predictions per section.",
        required=False,
        default="biotriplex_test_gold_entities.jsonl",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        required=False,
        default=0.3,
        help="Seconds to sleep between different PMC calls (politeness).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=True,
        help="Verbose diagnostics (requests, responses, context).",
    )
    return parser.parse_args()


# ------------------------- GOLD FROM XML ------------------------- #

_PMC_NUM_RE = re.compile(r"(\d+)")
_PMC_PREFIX_RE = re.compile(r"PMC(\d+)", re.IGNORECASE)

def normalize_pmc_numeric(raw: str) -> str:
    if not raw:
        raise ValueError("empty id")
    s = str(raw).strip()
    m = _PMC_PREFIX_RE.search(s)
    if m:
        return m.group(1)
    m2 = _PMC_NUM_RE.search(s)
    if m2:
        return m2.group(1)
    raise ValueError("cannot extract numeric pmc from: " + repr(raw))

def extract_pmids_from_test(test_path: str) -> Set[str]:
    pmids: Set[str] = set()
    if not os.path.isfile(test_path):
        raise FileNotFoundError("Test file not found: " + test_path)
    with open(test_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            m = _PMC_PREFIX_RE.search(s)
            if not m:
                m2 = _PMC_NUM_RE.search(s)
                if not m2:
                    raise RuntimeError(f"Line {i} has no PMC id: {s}")
                pmc_num = m2.group(1)
            else:
                pmc_num = m.group(1)
            pmids.add(pmc_num)
    if not pmids:
        raise RuntimeError("No PMC ids found in test file: " + test_path)
    return pmids

def parse_spans(spans_str: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for part in spans_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            s_str, e_str = part.split("-", 1)
        elif ":" in part:
            s_str, e_str = part.split(":", 1)
        else:
            bits = part.split()
            if len(bits) == 2:
                s_str, e_str = bits[0], bits[1]
            else:
                continue
        s, e = int(s_str), int(e_str)
        if s < e:
            spans.append((s, e))
    return spans

def extract_entities_from_xml(xml_path: str) -> Tuple[str, List[List]]:
    if not os.path.isfile(xml_path):
        raise FileNotFoundError("Gold XML not found: " + xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    text_el = root.find(".//TEXT")
    text = text_el.text if (text_el is not None and text_el.text is not None) else ""
    tags_el = root.find(".//TAGS")

    entities: List[List] = []
    if tags_el is None:
        raise RuntimeError("XML missing <TAGS>: " + xml_path)

    for g in tags_el.findall("GENE"):
        spans_attr = g.attrib.get("spans", "")
        for s, e in parse_spans(spans_attr):
            if s < 0 or e > len(text) or s >= e:
                raise RuntimeError(f"Bad GENE span {s}-{e} in {xml_path} (len={len(text)})")
            entities.append([s, e, "GENE"])
    for d in tags_el.findall("DISEASE"):
        spans_attr = d.attrib.get("spans", "")
        for s, e in parse_spans(spans_attr):
            if s < 0 or e > len(text) or s >= e:
                raise RuntimeError(f"Bad DISEASE span {s}-{e} in {xml_path} (len={len(text)})")
            entities.append([s, e, "DISEASE"])

    # dedupe and sort
    entities = sorted(set(tuple(e) for e in entities), key=lambda x: (x[0], x[1], x[2]))
    return text, [list(e) for e in entities]


# ------------------------- Strict enumeration ------------------------- #

def enumerate_sections(pmids_numeric: Set[str], xml_root: str) -> List[Tuple[str, str, str]]:
    # Returns list of (pmc_numeric, section_name, xml_path)
    sections: List[Tuple[str, str, str]] = []

    if not os.path.isdir(xml_root):
        raise NotADirectoryError("XML root not found: " + xml_root)

    for pmc_numeric in sorted(pmids_numeric, key=lambda x: int(x)):
        dir_candidates = [
            os.path.join(xml_root, "PMC" + pmc_numeric),
            os.path.join(xml_root, pmc_numeric),
        ]
        pmc_dir = next((d for d in dir_candidates if os.path.isdir(d)), None)
        if not pmc_dir:
            raise RuntimeError("No directory for pmc {} at {}".format(pmc_numeric, " or ".join(dir_candidates)))

        xml_files = [fn for fn in os.listdir(pmc_dir) if fn.lower().endswith(".xml")]
        if not xml_files:
            raise RuntimeError("No XML files found in {}".format(pmc_dir))

        matched = 0
        for fn in sorted(xml_files):
            valid_prefixes = ["PMC" + pmc_numeric + "_", pmc_numeric + "_"]
            prefix = next((p for p in valid_prefixes if fn.startswith(p)), None)
            if not prefix:
                continue
            section = fn[len(prefix):-4]
            xml_path = os.path.join(pmc_dir, fn)
            sections.append((pmc_numeric, section, xml_path))
            matched += 1

        if matched == 0:
            raise RuntimeError("XML files in {} do not match pmc {}".format(pmc_dir, pmc_numeric))

    if not sections:
        raise RuntimeError("No sections to process.")
    return sections


# ------------------------- PubTator3 HTTP ------------------------- #

def _http_get_json_with_retry(url: str, params: Dict[str, str], context: str) -> Any:
    # Retry HTTP 404 forever; crash on any other status or parse error.
    backoff = 1.5
    max_backoff = 60.0

    while True:
        try:
            full_url = requests.Request("GET", url, params=params).prepare().url
        except Exception:
            full_url = url
        debug_print(f"[pubtator3][export] {context} GET {full_url}")

        try:
            resp = requests.get(url, params=params, timeout=HTTP_TIMEOUT_SEC)
            debug_print(f"[pubtator3][export] {context} -> HTTP {resp.status_code} {resp.url}")
        except Exception as e:
            raise RuntimeError(f"HTTP request failed [{context}]: {repr(e)} URL(prepared)={full_url}") from e

        if resp.status_code == 200:
            try:
                return resp.json()
            except Exception as e:
                body_preview = (resp.text or "")[:400]
                raise RuntimeError(f"JSON parse failed [{context}]: {repr(e)} Body[:400]={repr(body_preview)} URL={resp.url}") from e

        if resp.status_code == 404:
            delay = backoff
            debug_print(f"[pubtator3][export] 404 not ready [{context}]; sleeping {delay:.2f}s then retry")
            time.sleep(delay)
            backoff = min(backoff * 1.5, max_backoff)
            continue

        body_preview = (resp.text or "")[:400]
        raise RuntimeError(f"Export HTTP {resp.status_code} [{context}] Body[:400]: {repr(body_preview)} URL: {resp.url}")

def _http_get_text_with_retry(url: str, params: Dict[str, str], context: str) -> str:
    # Retry HTTP 404 forever; crash on any other status.
    backoff = 1.5
    max_backoff = 60.0

    while True:
        try:
            full_url = requests.Request("GET", url, params=params).prepare().url
        except Exception:
            full_url = url
        debug_print(f"[pubtator3][export] {context} GET {full_url}")

        try:
            resp = requests.get(url, params=params, timeout=HTTP_TIMEOUT_SEC)
            debug_print(f"[pubtator3][export] {context} -> HTTP {resp.status_code} {resp.url}")
        except Exception as e:
            raise RuntimeError(f"HTTP request failed [{context}]: {repr(e)} URL(prepared)={full_url}") from e

        if resp.status_code == 200:
            return resp.text

        if resp.status_code == 404:
            delay = backoff
            debug_print(f"[pubtator3][export] 404 not ready [{context}]; sleeping {delay:.2f}s then retry")
            time.sleep(delay)
            backoff = min(backoff * 1.5, max_backoff)
            continue

        body_preview = (resp.text or "")[:400]
        raise RuntimeError(f"Export HTTP {resp.status_code} [{context}] Body[:400]: {repr(body_preview)} URL: {resp.url}")


# ------------------------- PubTator3 fetchers ------------------------- #

def pmc_numeric_to_pmid(pmc_numeric: str) -> str:
    key = str(pmc_numeric).strip()
    if key in _PMC2PMID_CACHE:
        return _PMC2PMID_CACHE[key]

    pmcid = "PMC" + key
    params = {"ids": pmcid, "format": "json"}
    data = _http_get_json_with_retry(PMC_IDCONV_URL, params, context=f"idconv {pmcid}")

    recs = (data or {}).get("records") or []
    if not recs:
        raise RuntimeError(f"No idconv records for {pmcid}")
    rec = recs[0]
    pmid = str(rec.get("pmid") or "").strip()
    if not pmid:
        raise RuntimeError(f"No PMID mapping found for {pmcid} (record={rec})")

    _PMC2PMID_CACHE[key] = pmid
    return pmid

def _normalize_bioc_documents_json(bioc: Any) -> List[dict]:
    # Accept {"PubTator3": [...]}, {"documents":[...]}, or list
    if isinstance(bioc, dict):
        if "PubTator3" in bioc and isinstance(bioc["PubTator3"], list):
            return bioc["PubTator3"]
        if "documents" in bioc and isinstance(bioc["documents"], list):
            return bioc["documents"]
    if isinstance(bioc, list):
        return bioc
    raise RuntimeError(f"Unexpected PubTator JSON envelope: {type(bioc)}")

def fetch_pubtator_publication_by_pmid(pmid: str) -> List[dict]:
    # Title/abstract by PMID (BioC JSON)
    key = str(pmid).strip()
    if key in _PUB_DOC_CACHE_BY_PMID:
        return _PUB_DOC_CACHE_BY_PMID[key]["__docs__"]

    url = f"{PUBTATOR3_PUBLICATIONS_EXPORT_BASE}/biocjson"
    params = {"pmids": key, "concepts": ",".join(CONCEPTS)}
    bioc = _http_get_json_with_retry(url, params, context=f"pmid={key}")
    docs = _normalize_bioc_documents_json(bioc)
    if not isinstance(docs, list) or not docs:
        raise RuntimeError(f"No documents for PMID {key} in PubTator JSON.")
    _PUB_DOC_CACHE_BY_PMID[key] = {"raw": bioc, "__docs__": docs}
    return docs

def _xml_get_text(el: Optional[ET.Element]) -> str:
    if el is None:
        return ""
    # Join all text nodes under this element
    return "".join(el.itertext())

def _biocxml_to_documents(xml_text: str) -> List[dict]:
    # Parse BioC XML into a normalized documents list
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        raise RuntimeError(f"BioC XML parse error: {repr(e)}") from e

    docs: List[dict] = []
    for doc in root.findall(".//document"):
        doc_id = _xml_get_text(doc.find("./id")) or ""
        passages: List[dict] = []
        for p in doc.findall("./passage"):
            # infons
            infons: Dict[str, str] = {}
            for inf in p.findall("./infon"):
                k = inf.attrib.get("key", "")
                v = _xml_get_text(inf)
                if k:
                    infons[k] = v
            text = _xml_get_text(p.find("./text")) or ""
            # passage offset (optional)
            p_offset_text = _xml_get_text(p.find("./offset")).strip() if p.find("./offset") is not None else "0"
            try:
                p_offset = int(p_offset_text or "0")
            except ValueError:
                raise RuntimeError(f"Non-integer passage offset: {p_offset_text}")
            # annotations
            ann_list: List[dict] = []
            for a in p.findall("./annotation"):
                a_infons: Dict[str, str] = {}
                for inf in a.findall("./infon"):
                    k = inf.attrib.get("key", "")
                    v = _xml_get_text(inf)
                    if k:
                        a_infons[k] = v
                locs = []
                for loc in a.findall("./location"):
                    off = loc.attrib.get("offset", "0")
                    leng = loc.attrib.get("length", "0")
                    try:
                        o = int(off)
                        L = int(leng)
                    except ValueError:
                        raise RuntimeError(f"Non-integer location offset/length: {off}/{leng}")
                    locs.append({"offset": o, "length": L})
                ann_list.append({"infons": a_infons, "locations": locs})
            passages.append({"infons": infons, "text": text, "annotations": ann_list, "offset": p_offset})
        docs.append({"id": doc_id, "passages": passages})

    if not docs:
        # Fallback: treat top-level passages if no documents found
        passages = []
        for p in root.findall(".//passage"):
            infons: Dict[str, str] = {}
            for inf in p.findall("./infon"):
                k = inf.attrib.get("key", "")
                v = _xml_get_text(inf)
                if k:
                    infons[k] = v
            text = _xml_get_text(p.find("./text")) or ""
            p_offset_text = _xml_get_text(p.find("./offset")).strip() if p.find("./offset") is not None else "0"
            try:
                p_offset = int(p_offset_text or "0")
            except ValueError:
                raise RuntimeError(f"Non-integer passage offset: {p_offset_text}")
            ann_list: List[dict] = []
            for a in p.findall("./annotation"):
                a_infons: Dict[str, str] = {}
                for inf in a.findall("./infon"):
                    k = inf.attrib.get("key", "")
                    v = _xml_get_text(inf)
                    if k:
                        a_infons[k] = v
                locs = []
                for loc in a.findall("./location"):
                    off = loc.attrib.get("offset", "0")
                    leng = loc.attrib.get("length", "0")
                    o = int(off); L = int(leng)
                    locs.append({"offset": o, "length": L})
                ann_list.append({"infons": a_infons, "locations": locs})
            if passages or text or infons:
                passages.append({"infons": infons, "text": text, "annotations": ann_list, "offset": p_offset})
        if passages:
            docs = [{"id": "", "passages": passages}]

    if not docs:
        raise RuntimeError("Empty BioC XML: no documents found.")
    return docs

def fetch_pubtator_pmc_publication_docs_by_pmcid_xml(pmc_numeric: str) -> List[dict]:
    """
    Fetch a BioC XML full-text document(s) from PubTator3 PMC publications export by PMCID.
    Example:
      https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/pmc_export/biocxml?pmcids=PMC<digits>
    """
    key = str(pmc_numeric).strip()
    if key in _PUB_DOC_CACHE_BY_PMCID_XMLDOCS:
        return _PUB_DOC_CACHE_BY_PMCID_XMLDOCS[key]

    pmcid = "PMC" + key
    url = f"{PUBTATOR3_PUBLICATIONS_PMC_EXPORT_BASE}/biocxml"
    params = {"pmcids": pmcid}
    xml_text = _http_get_text_with_retry(url, params, context=f"pmcid={pmcid}")
    docs = _biocxml_to_documents(xml_text)
    _PUB_DOC_CACHE_BY_PMCID_XMLDOCS[key] = docs
    return docs


# ------------------------- Parsing predictions ------------------------- #

def _list_available_sections_in_docs(documents: List[dict]) -> List[str]:
    seen: List[str] = []
    for doc in documents:
        for p in (doc.get("passages") or []):
            infons = p.get("infons") or {}
            c = passage_section_canonical(infons)
            if c and c not in seen:
                seen.append(c)
    return seen

def _parse_section_entities_from_docs(documents: List[dict], pmc_numeric: str, target_section_label: str, target_canon: str, text_len: int) -> List[List]:
    # List available sections for diagnostics
    avail = _list_available_sections_in_docs(documents)
    debug_print(f"[sections] want={target_section_label} canon={target_canon} available={avail}")

    if target_canon not in avail:
        raise RuntimeError(
            f"Section not found for PMC{pmc_numeric}: want={target_section_label} canon={target_canon} available={avail}"
        )

    # Collect passages with the exact canonical section
    matching_passages = []
    for doc in documents:
        for p in (doc.get("passages") or []):
            infons = p.get("infons") or {}
            canon = passage_section_canonical(infons)
            if canon == target_canon:
                matching_passages.append(p)

    if not matching_passages:
        # Should not happen because of the avail check, but keep strict
        raise RuntimeError(
            f"No passages for section PMC{pmc_numeric} {target_canon}. Available={avail}"
        )

    if len(matching_passages) > 1:
        # Strict: ambiguous alignment with gold single-section text
        details = [(pp.get("offset", 0), len(pp.get("text") or "")) for pp in matching_passages]
        raise RuntimeError(
            f"Multiple passages for section PMC{pmc_numeric} {target_canon}: {details}. "
            f"Ambiguous alignment vs. single gold XML section."
        )

    p = matching_passages[0]
    p_text = p.get("text") or ""
    p_offset = int(p.get("offset") or 0)

    # Strict: the PubTator section text must match the gold TEXT length
    if len(p_text) != text_len:
        raise RuntimeError(
            f"Text length mismatch for PMC{pmc_numeric} {target_canon}: PubTator len={len(p_text)} vs gold len={text_len}"
        )

    entities: List[List] = []
    for ann in (p.get("annotations") or []):
        a_inf = ann.get("infons") or {}
        a_type = str(a_inf.get("type") or "").strip().upper()
        if a_type not in ("GENE", "DISEASE"):
            continue
        for loc in (ann.get("locations") or []):
            o_abs = int(loc.get("offset", 0))
            L = int(loc.get("length", 0))
            start = o_abs - p_offset
            end = start + L
            # Strict range checks
            if start < 0 or end > len(p_text) or start >= end:
                raise RuntimeError(
                    f"Bad annotation span PMC{pmc_numeric} {target_canon}: "
                    f"loc(offset={o_abs},len={L}) passage_offset={p_offset} -> local {start}-{end} out of [0,{len(p_text)}]"
                )
            entities.append([start, end, a_type])

    # dedupe and sort
    entities = sorted(set(tuple(e) for e in entities), key=lambda x: (x[0], x[1], x[2]))
    return [list(e) for e in entities]

def get_pubtator_predictions_for_section(pmc_numeric: str, section: str, section_text: str) -> List[List]:
    target_canon = canonicalize_section_name(section)
    if target_canon in ("TITLE", "ABSTRACT"):
        pmid = pmc_numeric_to_pmid(pmc_numeric)
        docs = fetch_pubtator_publication_by_pmid(pmid)
        # For PubMed JSON, we need to synthesize a section view: ensure exactly one passage of that type
        # Normalize documents into one pseudo-document containing only the target passage
        passages: List[dict] = []
        for doc in docs:
            for p in (doc.get("passages") or []):
                infons = p.get("infons") or {}
                canon = passage_section_canonical(infons)
                if not canon:
                    t = str(infons.get("type") or "").strip().lower()
                    if t == "title":
                        canon = "TITLE"
                    elif t.startswith("abstract"):
                        canon = "ABSTRACT"
                if canon == target_canon:
                    # PubTator JSON includes passage "offset"; if missing, treat as 0
                    p_offset = int(p.get("offset", 0)) if isinstance(p.get("offset", 0), int) else int(p.get("offset", 0) or 0)
                    passages.append({
                        "infons": infons,
                        "text": p.get("text") or "",
                        "annotations": p.get("annotations") or [],
                        "offset": p_offset
                    })
        if not passages:
            avail = _list_available_sections_in_docs(docs)
            raise RuntimeError(f"Section not found for PMC{pmc_numeric} via PMID {pmid}: want={section} canon={target_canon} available={avail}")
        if len(passages) > 1:
            details = [(pp.get("offset", 0), len(pp.get("text") or "")) for pp in passages]
            raise RuntimeError(f"Multiple {target_canon} passages for PMC{pmc_numeric}/PMID {pmid}: {details}")
        documents = [{"id": "", "passages": passages}]
    else:
        documents = fetch_pubtator_pmc_publication_docs_by_pmcid_xml(pmc_numeric)

    return _parse_section_entities_from_docs(
        documents,
        pmc_numeric=pmc_numeric,
        target_section_label=section,
        target_canon=target_canon,
        text_len=len(section_text),
    )


# ------------------------- Scoring ------------------------- #

def score_entities(gold: List[List], pred: List[List]) -> Dict[str, int]:
    gold_set = set(tuple(e) for e in gold)
    pred_set = set(tuple(e) for e in pred)

    tp_all = len(gold_set & pred_set)
    fp_all = len(pred_set - gold_set)
    fn_all = len(gold_set - pred_set)

    gold_gene = {e for e in gold_set if e[2] == "GENE"}
    gold_dis = {e for e in gold_set if e[2] == "DISEASE"}
    pred_gene = {e for e in pred_set if e[2] == "GENE"}
    pred_dis = {e for e in pred_set if e[2] == "DISEASE"}

    tp_gene = len(gold_gene & pred_gene)
    fp_gene = len(pred_gene - gold_gene)
    fn_gene = len(gold_gene - pred_gene)

    tp_dis = len(gold_dis & pred_dis)
    fp_dis = len(pred_dis - gold_dis)
    fn_dis = len(gold_dis - pred_dis)

    return {
        "tp_all": tp_all, "fp_all": fp_all, "fn_all": fn_all,
        "tp_gene": tp_gene, "fp_gene": fp_gene, "fn_gene": fn_gene,
        "tp_dis": tp_dis, "fp_dis": fp_dis, "fn_dis": fn_dis,
    }

def prec_rec_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


# ------------------------- Main processing ------------------------- #

def process_pmids(pmids_numeric: Set[str], xml_root: str, output_gold: str, sleep: float) -> None:
    sections = enumerate_sections(pmids_numeric, xml_root)
    total_planned = len(sections)
    sys.stderr.write("Planned sections to process: " + str(total_planned) + "\n")

    tp_all = fp_all = fn_all = 0
    tp_gene = fp_gene = fn_gene = 0
    tp_dis = fp_dis = fn_dis = 0

    with open(output_gold, "w", encoding="utf-8") as fout:
        last_pmc_numeric: Optional[str] = None
        for idx, (pmc_numeric, section, xml_path) in enumerate(sections, start=1):
            if pmc_numeric != last_pmc_numeric and last_pmc_numeric is not None and sleep > 0:
                time.sleep(sleep)
            last_pmc_numeric = pmc_numeric

            canon_section = canonicalize_section_name(section)
            sys.stderr.write(f"Progress: {idx}/{total_planned} ({idx*100.0/total_planned:.1f}%) - pmc {pmc_numeric} [{canon_section}]\n")
            sys.stderr.flush()

            text, gold_ents = extract_entities_from_xml(xml_path)

            pred_ents = get_pubtator_predictions_for_section(pmc_numeric, canon_section, text)

            sc = score_entities(gold_ents, pred_ents)
            tp_all += sc["tp_all"]; fp_all += sc["fp_all"]; fn_all += sc["fn_all"]
            tp_gene += sc["tp_gene"]; fp_gene += sc["fp_gene"]; fn_gene += sc["fn_gene"]
            tp_dis += sc["tp_dis"]; fp_dis += sc["fp_dis"]; fn_dis += sc["fn_dis"]

            out = {
                "pmc_numeric": pmc_numeric,
                "section": canon_section,
                "xml_path": xml_path,
                "gold_entities": gold_ents,
                "pred_entities": pred_ents,
                "stats": sc,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    sys.stderr.write("\n")

    p_all, r_all, f_all = prec_rec_f1(tp_all, fp_all, fn_all)
    p_gene, r_gene, f_gene = prec_rec_f1(tp_gene, fp_gene, fn_gene)
    p_dis, r_dis, f_dis = prec_rec_f1(tp_dis, fp_dis, fn_dis)

    print("== PubTator3 vs. XML gold (GENE + DISEASE) ==")
    print("TP=" + str(tp_all) + " FP=" + str(fp_all) + " FN=" + str(fn_all))
    print("Precision: {:.4f}".format(p_all))
    print("Recall:    {:.4f}".format(r_all))
    print("F1:        {:.4f}\n".format(f_all))

    print("== GENE only ==")
    print("TP=" + str(tp_gene) + " FP=" + str(fp_gene) + " FN=" + str(fn_gene))
    print("Precision: {:.4f}".format(p_gene))
    print("Recall:    {:.4f}".format(r_gene))
    print("F1:        {:.4f}\n".format(f_gene))

    print("== DISEASE only ==")
    print("TP=" + str(tp_dis) + " FP=" + str(fp_dis) + " FN=" + str(fn_dis))
    print("Precision: {:.4f}".format(p_dis))
    print("Recall:    {:.4f}".format(r_dis))
    print("F1:        {:.4f}".format(f_dis))


def main() -> None:
    args = parse_args()
    global DEBUG
    DEBUG = args.debug

    # Environment banner
    req_ver = getattr(requests, "__version__", "unknown")
    debug_print("=== Environment ===")
    debug_print("python: " + platform.python_version() + " on " + platform.system() + " " + platform.release())
    debug_print("requests: " + req_ver)
    debug_print("publications export: " + PUBTATOR3_PUBLICATIONS_EXPORT_BASE + "/{biocjson}?pmids=<digits>&concepts=gene,disease")
    debug_print("pmc publications export: " + PUBTATOR3_PUBLICATIONS_PMC_EXPORT_BASE + "/{biocxml}?pmcids=PMC<digits>")

    pmids_numeric = extract_pmids_from_test(args.test_jsonl)
    sys.stderr.write("Found " + str(len(pmids_numeric)) + " PMC numeric codes in test split.\n")
    process_pmids(pmids_numeric=pmids_numeric, xml_root=args.xml_root, output_gold=args.output_gold, sleep=args.sleep)


if __name__ == "__main__":
    main()
