# Student implementation projects

These projects are designed as independent modules. No implementation has been
provided so students can design, test, and document their own solutions.

## Project 1: PMC acquisition and provenance

Build a reproducible collector for PMC Open Access figures and metadata.

Deliverables:

- query configuration;
- rate-limited acquisition;
- article, figure, caption, and license provenance;
- resumable downloads;
- checksums and structured logs;
- synthetic tests.

## Project 2: Figure relevance classifier

Classify figures as diagnostically relevant or irrelevant for breast cancer
and brain tumors.

Deliverables:

- labelling protocol;
- train/validation/test split at article level;
- baseline and error analysis;
- calibration and abstention;
- confusion matrix by figure type.

## Project 3: Multi-panel figure decomposition

Detect and separate compound publication figures before localization.

Deliverables:

- panel-boundary detection;
- caption-panel alignment;
- comparison with unseparated figures;
- failure taxonomy;
- preservation of source provenance.

## Project 4: Caption-based disease labelling

Replace the keyword heuristic with a validated classifier.

Deliverables:

- explicit label ontology;
- uncertainty and ambiguous class;
- clinician-reviewed evaluation subset;
- precision, recall, calibration, and subgroup analysis;
- comparison with keyword rules.

## Project 5: Text-grounded localization

Generate lesion boxes from figure captions and images.

Deliverables:

- prompt construction;
- box-confidence scores;
- fallback policy;
- localization metrics;
- comparison against manually reviewed boxes.

## Project 6: Segmentation benchmark

Compare MCP-MedSAM Lite, Swift-MedSAM, and base MedSAM on the same prompts.

Deliverables:

- fixed inputs and deterministic configuration;
- IoU and boundary metrics;
- latency and memory measurements;
- per-modality analysis;
- prompt-provenance stratification.

## Project 7: Representation-audit toolkit

Automate dataset composition and confounding checks.

Deliverables:

- modality and disease distributions;
- under-representation thresholds;
- cross-tabulations;
- minimum-sample warnings;
- machine-readable and human-readable reports.

## Project 8: Failure-review interface

Create a local interface for reviewing image, prompt, mask, and failure flags.

Deliverables:

- no transmission of images to third-party services;
- reviewer notes and adjudication;
- blinded sampling;
- exportable corrections;
- audit log.

## Shared acceptance criteria

Every student project should:

- use only authorized data;
- include tests and documentation;
- preserve provenance;
- avoid article-level leakage;
- separate exploratory from confirmatory analysis;
- report uncertainty and failure modes;
- never claim clinical readiness; and
- submit work through a reviewed pull request.

