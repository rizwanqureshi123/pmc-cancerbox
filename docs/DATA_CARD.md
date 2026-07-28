# PMC-CancerBox dataset card

## Dataset summary

PMC-CancerBox is a scientific-literature-derived medical imaging corpus for
breast-cancer and brain-tumor segmentation research. The reported release
contains 9,345 image-mask pairs associated with 9,319 PubMed Central figure
captions.

## Intended uses

- Research on weakly supervised and prompt-guided segmentation
- Evaluation of medical-image grounding
- Multi-panel figure decomposition
- Caption-based relevance and disease classification
- Dataset representation and failure-mode auditing
- Education in reproducible and responsible medical AI

## Out-of-scope uses

- Clinical diagnosis, treatment, or triage
- Estimating performance in a consecutive clinical population
- Demographic fairness assessment
- Claims of generalization to modalities with very small samples
- Patient-level analysis

## Source and construction

The reported pipeline queries PubMed Central Open Access, retrieves figures and
metadata, filters images for diagnostic relevance, creates bounding-box
prompts, and uses MCP-MedSAM to produce segmentation masks without retraining.

## Size

| Component | Count |
|---|---:|
| Image-mask pairs | 9,345 |
| Figure captions | 9,319 |
| Unique captions | 8,662 |
| Dated source articles | 4,182 |
| Unique caption tokens | 16,734 |

## Modality distribution

| Modality | Records | Share | Mean IoU |
|---|---:|---:|---:|
| MR | 4,077 | 43.6% | 0.4634 |
| CT | 3,861 | 41.3% | 0.5885 |
| Ultrasound | 529 | 5.7% | 0.7233 |
| Mammography | 421 | 4.5% | 0.5752 |
| Microscopy | 298 | 3.2% | 0.7232 |
| PET | 85 | 0.9% | 0.6311 |
| X-ray | 66 | 0.7% | 0.5800 |
| Fundus | 5 | 0.1% | 0.9047 |
| Endoscopy | 2 | <0.1% | 0.8606 |
| OCT | 1 | <0.1% | 0.5865 |

Rows with fewer than 30 records are descriptive only.

## Disease labels

Caption-keyword heuristics assigned:

| Label | Records | Share |
|---|---:|---:|
| Breast cancer | 2,567 | 27.5% |
| Brain tumor | 2,259 | 24.2% |
| Both or ambiguous | 30 | 0.3% |
| Other | 4,489 | 48.0% |

These are weak corpus labels and must not be treated as clinician-adjudicated
diagnoses.

## Caption characteristics

- Mean length: 55.27 words
- Duplicate rate: 7.05%
- Captions containing at least one medical term: 91.95%
- Panel labels and MRI sequence identifiers create systematic tokenization
  artifacts.

## Known limitations

1. MR and CT account for 84.9% of records.
2. Radiological images account for 96.8%.
3. Modality and disease are confounded.
4. Patient demographics are unavailable.
5. Published figures are selected for illustrative clarity.
6. Multi-panel layouts are common and can be mistaken for multiple lesions.
7. Disease labels are derived from caption keywords.
8. Duplicate captions should be controlled before train/test splitting.
9. Article recency produces temporal and technology-selection bias.

## Release checklist

Before public data release, add:

- stable archive URL and version;
- file-level checksums;
- complete schema documentation;
- license and source-license mapping;
- inclusion and exclusion rules;
- split-generation procedure;
- duplicate and leakage controls;
- model-output provenance;
- takedown and correction process; and
- a changelog.

