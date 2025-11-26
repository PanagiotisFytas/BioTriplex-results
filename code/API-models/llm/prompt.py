basic_prompt = """
### Overview
The aim of the task is to annotate texts by marking words and phrases (‘entities’) that relate to human diseases, genes and the relationships between them.
Markable entities correspond to one of three defined categories: "Human Diseases", "Genes", and "Relations".

### Entities
- **Human Diseases**: Any disease or disorder that affects the human body.
- **Genes**: Any human gene names, symbols or synonyms.
- **Relations**: Relationships between a human disease and a gene. The following relation types are defined:
    - **no relation**: No relationship between the disease and gene.
    - **relation undefined**: Relationship between the disease and gene is not defined.
    - **pathological role**: The gene plays a role in the disease.
    - **causative activation**: The gene activates the disease.
    - **causative inhibition**: The gene inhibits the disease.
    - **causative mutation**: The gene causes a mutation in the disease.
    - **modulator decrease disease**: The gene decreases the disease.
    - **modulator increase disease**: The gene increases the disease.
    - **biomarker**: The gene is a biomarker for the disease.
    - **associated mutation**: The gene is associated with a mutation in the disease.
    - **dysregulation**: The gene is dysregulated in the disease.
    - **increased expression**: The gene is expressed at higher levels in the disease.
    - **decreased expression**: The gene is expressed at lower levels in the disease.
    - **epigenetic marker**: The gene is an epigenetic marker for the disease.
    - **therapy resistance**: The gene is associated with therapy resistance in the disease.
    - **prognostic indicator**: The gene is a prognostic indicator for the disease.
    - **negative prognostic marker**: The gene is a negative prognostic marker for the disease.
    - **positive prognostic marker**: The gene is a positive prognostic marker for the disease.
    - **therapeutic target**: The gene is a therapeutic target for the disease.
    - **diagnostic tool**: The gene is a diagnostic tool for the disease.
    - **genetic susceptibility**: The gene is associated with genetic susceptibility to the disease.
    
### Examples of entities
- **Human Diseases**: "lung adenocarcinoma", "atherosclerotic cardiovascular disease", "Loeys-Dietz syndrome".
- **Genes**: "SLC02A1", "PCSK5", "Angiotensin I converting enzyme".
- **Relations**:
    - **increased expression**: "highly expressed".
    - **therapy resistance**: "drug resistance".
    - **causative mutation**: "causative mutation".
    
### Annotation Guidelines
- General guidelines:
    - Mark every example of each Gene and Human Disease which occurs in the text.
    - Mark every example of each Relation which describes the relationship between a marked Gene and a marked Human Disease.
    - Mark the whole word or phrase and do not include any blank space or punctuation (e.g. brackets) outside the boundaries.
    - In a few cases, the spans of entities may overlap with others.
    - Do not mark extraneous text. Mark multi-span entities (entities composed of two or more text segments) where necessary. 
    
- Guidelines for Human Diseases:
    - Mark every example of a full Disease name that corresponds to a disease.
    - Mark every abbreviation of listed Diseases.
    - Disease abbreviations and long forms should be annotated as separate entities. For example, in the phrase "Duchenne Muscular Dystrophy (DMD)", "Duchenne Muscular Dystrophy" and "DMD" should each be marked as a separate entity and will each correspond to the same human disease identifier.

- Guidelines for Genes:
    - Mark every example of a human gene (symbol, name or synonym) that corresponds to a human gene.
    - Do not mark genes stated to be that of any other non-human species.
    - Do not mark incomplete names of genes or the names of gene families e.g. mark "WNT1"; do not mark "Wnt family genes".
    - Do not mark genes that are named as part of the name or description of a biochemical mechanism or pathway e.g. in "Wnt/Beta-catenin pathway" do not mark "Beta-catenin".

- Guidelines for Relations:
    - Mark Relations (words and noun phrases) that correspond to the relation types listed above.
    - Describe the relationship between the marked Gene and marked Human Disease.
    - Words and phrases that have different surface forms but which correspond to the sense of the listed Relation types should be marked.
    - Do not mark words or noun phrases that describe the relation between any other pairs of entities (e.g. between a gene and another gene) or between a marked entity and an unmarked word or phrase. 
    - Do not mark words or phrases that correspond to a listed relation type but which do not describe the relationship between a marked Gene and a marked Human Disease.
   
### Output Format
- The output file should be in the JSON format.
- The JSON file should contain a list of dictionaries, where each dictionary corresponds to a sentence in the input text.
- Each dictionary should have the following
    - "text": The text of the sentence.
    - "entities": A list of dictionaries, where each dictionary corresponds to an entity in the sentence.
        - Each entity dictionary should have the following keys:
            - "start": The start index of the entity.
            - "end": The end index of the entity.
            - "label": The label of the entity.
    - "relations": A list of dictionaries, where each dictionary corresponds to a relation in the sentence.
        - Each relation dictionary should have the following keys:
            - "entity1": The index of the first entity in the relation.
            - "entity2": The index of the second entity in the relation.
            - "label": The label of the relation.
    
"""


# now modify this prompt for triplet extraction instead of NER and RE

triplet_prompt =\
"""
You are tasked with extracting triplets from biomedical text. Each triplet consists of three linked entities: a gene, a human disease, and a relation between them. Follow the instructions below to extract these triplets.

### Task:
**Extract triplets**: Identify and extract sets of three linked entities:
   - **Gene**: A human gene name, symbol (e.g., *SLC02A1*, *PCSK5*) or synonym.
   - **Human Disease**: A specific human disease or disorder name (e.g., *lung adenocarcinoma*, *coronary artery disease*).
   - **Relation**: The type of relationship between the gene and the human disease. These relations of interest are *pathological role*, *causative activation*, *causative inhibition*, *causative mutation*, *modulator decrease disease*, *modulator increase disease*, *biomarker*, *associated mutation*, *dysregulation*, *increased expression*, *decreased expression*, *epigenetic marker*, *therapy resistance*, *prognostic indicator*, *negative prognostic marker*, *positive prognostic marker*, *therapeutic target*, *diagnostic tool*, *genetic susceptibility*.

### Examples:

#### Example 1:
**Text:** 
"Our finding suggested SETD2 as a potential epigenetic marker in LUAD patients."
**Output:**
[{ "Gene": "SETD2", "Relation": "epigenetic marker", "Human Disease": "LUAD" }]

#### Example 2:
**Text:**
"CDKN2BAS1/ANRIL , located in the 9p21 chromosomic region, has been reported in numerous studies as a genetic risk locus for CAD, intracranial aneurysms and diverse cardiometabolic disorders."
** Output:**
[{ "Gene": "CDKN2BAS1", "Relation": "genetic risk locus", "Human Disease": "CAD" }, { "Gene": "CDKN2BAS1", "Relation": "genetic risk locus", "Human Disease": "intracranial aneurysms" }]

#### Example 3:
**Text:** 
"Duchenne muscular dystrophy (DMD) is an X-linked inherited neuromuscular disorder due to mutations in the dystrophin gene."
**Output:**
[{ "Gene": "dystrophin", "Relation": "due to mutations", "Human Disease": "Duchenne muscular dystrophy" }, { "Gene": "dystrophin", "Relation": "due to mutations", "Human Disease": "DMD" }]

#### Example 4:
**Text:**
"activation of Wnt signalling is nonetheless thought to play an important role in breast tumorigenesis."
**Output:**
[]

#### Example 5:
**Text:**
"high expression of METTL3, HNRNPA2B1, and YTHDF3 were related to the poor prognosis of osteosarcoma."
**Output:** [{ "Gene": "METTL3", "Relation": "high expression", "Human Disease": "osteosarcoma" }, { "Gene": "HNRNPA2B1", "Relation": "high expression", "Human Disease": "osteosarcoma" }, { "Gene": "YTHDF3", "Relation": "high expression", "Human Disease": "osteosarcoma" }, { "Gene": "METTL3", "Relation": "poor prognosis", "Human Disease": "osteosarcoma" }, { "Gene": "HNRNPA2B1", "Relation": "poor prognosis", "Human Disease": "osteosarcoma" }, { "Gene": "YTHDF3", "Relation": "poor prognosis", "Human Disease": "osteosarcoma" }]

#### Example 6:
**Text:**
"Title: Proteomic Analyses Identify Therapeutic Targets in Hepatocellular Carcinoma.\nAbstract: Hepatocellular carcinoma (HCC) is the fourth cause of cancer-related mortality worldwide. While many targeted therapies have been developed, the majority of HCC tumors do not harbor clinically actionable mutations. Protein-level aberrations, especially those not evident at the genomic level, present therapeutic opportunities but have rarely been systematically characterized in HCC. In this study, we performed proteogenomic analyses of 260 primary tumors from two HBV-related HCC patient cohorts with global mass-spectrometry (MS) proteomics data. Combining tumor-normal and inter-tumor analyses, we identified overexpressed targets including PDGFRB, FGFR4, ERBB2/3, CDK6 kinases and MFAP5, HMCN1, and Hsp proteins in HCC, many of which showed low frequencies of genomic and/or transcriptomic aberrations. Protein expression of FGFR4 kinase and Hsp proteins were significantly associated with response to their corresponding inhibitors. Our results provide a catalog of protein targets in HCC and demonstrate the potential of proteomics approaches in advancing precision medicine in cancer types lacking druggable mutations."
** Output:**
[{ "Gene": "PDGFRB", "Relation": "overexpressed targets", "Human Disease": "HCC" }, { "Gene": "FGFR4", "Relation": "overexpressed targets", "Human Disease": "HCC" }, { "Gene": "ERBB2", "Relation": "overexpressed targets", "Human Disease": "HCC" }, { "Gene": "CDK6", "Relation": "overexpressed targets", "Human Disease": "HCC" }, { "Gene": "MFAP5", "Relation": "overexpressed targets", "Human Disease": "HCC" }, { "Gene": "HMCN1", "Relation": "overexpressed targets", "Human Disease": "HCC" }]

### Guidelines:
- Extract **only complete triplets** where a gene, a human disease, and a relation are clearly linked.
- Ignore any entities or relations that do not form a complete triplet.
- Abbreviations of disease names (e.g., *DMD* for *Duchenne muscular dystrophy*) and gene synonyms should be handled as separate entities.
- Ignore gene from non-human species.
- Ignore incomplete names of genes and gene families (e.g., *Wnt family genes*).
- Ignore gene names that are part of the name or description of a biochemical mechanism or pathway (e.g., *Wnt/Beta-catenin pathway*).
- Extract relations from the specific relationship types listed above. Words or phrases that correspond to the sense of the listed relation types should be extracted, even if they have different surface forms.
- Do not extract relations that describe the relationship between any other pairs of entities (e.g., between a gene and another gene) or between a marked entity and an unmarked word or phrase.
- Do not extract relations that do not describe the relationship between a marked gene and a marked human disease.
- Extract only explict relations between a gene and a human disease. Do not extract implicit relations.

### Output Format:
- Provide the extracted triplets in the following json format:
    [{ "Gene": **Gene**, "Relation": **Relation**, "Human Disease": **Human Disease** }, ...]
- Each triplet should be a json dictionary with the keys "Gene", "Relation", and "Human Disease" and the corresponding values the extracted text for each entity.
- If no triplets are found in the text, provide an empty list.
- Do not generate any additional output and strictly adhere to the output format.

### Text for Extraction:
"""

sample_input =\
"""Title: EGFR activation triggers cellular hypertrophy and lysosomal disease in NAGLU-depleted cardiomyoblasts, mimicking the hallmarks of mucopolysaccharidosis IIIB.\nAbstract: Mucopolysaccharidosis (MPS) IIIB is an inherited lysosomal storage disease caused by the deficiency of the enzyme α- N -acetylglucosaminidase (NAGLU) required for heparan sulfate (HS) degradation. The defective lysosomal clearance of undigested HS results in dysfunction of multiple tissues and organs. We recently demonstrated that the murine model of MPS IIIB develops cardiac disease, valvular abnormalities, and ultimately heart failure. To address the molecular mechanisms governing cardiac dysfunctions in MPS IIIB, we generated a model of the disease by silencing NAGLU gene expression in H9C2 rat cardiomyoblasts. NAGLU-depleted H9C2 exhibited accumulation of abnormal lysosomes and a hypertrophic phenotype. Furthermore, we found the specific activation of the epidermal growth factor receptor (EGFR), and increased phosphorylation levels of extracellular signal-regulated kinases (ERKs) in NAGLU-depleted H9C2. The inhibition of either EGFR or ERKs, using the selective inhibitors AG1478 and PD98059, resulted in the reduction of both lysosomal aberration and hypertrophy in NAGLU-depleted H9C2. We also found increased phosphorylation of c-Src and a reduction of the hypertrophic response in NAGLU-depleted H9C2 transfected with a dominant-negative c-Src. However, c-Src phosphorylation remained unaffected by AG1478 treatment, posing c-Src upstream EGFR activation. Finally, heparin-binding EGF-like growth factor (HB-EGF) protein was found overexpressed in our MPS IIIB cellular model, and its silencing reduced the hypertrophic response. These results indicate that both c-Src and HB-EGF contribute to the hypertrophic phenotype of NAGLU-depleted cardiomyoblasts by synergistically activating EGFR and subsequent signaling, thus suggesting that EGFR pathway inhibition could represent an effective therapeutic approach for MPS IIIB cardiac disease.\n
"""

system_prompt = "You are tasked with extracting triplets from biomedical text. Each triplet consists of three linked entities: a gene, a human disease, and a relation between them. Follow the instructions closely below to extract these triplets. Output the extracted triplets in the specified json format. Adhere to the guidelines and the output format strictly. Do not generate any additional output.\n"