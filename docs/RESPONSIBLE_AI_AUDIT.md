# Responsible-AI representation audit

The representation audit is a core contribution of PMC-CancerBox rather than
an appendix to the segmentation results.

## Main findings

### Modality imbalance

MR and CT make up 84.9% of the corpus. Seven modalities fall below the 5%
under-representation threshold. Any model trained naively on the dataset will
behave primarily as an MR/CT model.

### Imaging-family imbalance

Radiology accounts for 96.8% of records, compared with 3.2% microscopy or
histopathology. The microscopy subset should be treated as exploratory.

### Disease-modality confounding

- 99.3% of mammography records are labelled breast cancer.
- 70.0% of brain-tumor records are MR.
- Breast-cancer records span a more diverse set of modalities.
- MR is both the largest modality and the weakest well-sampled modality by
  mean IoU.

Aggregate comparisons between diseases therefore confound clinical condition
with imaging modality.

### Missing demographic information

PMC figure metadata do not provide the patient-level demographic attributes
required for a valid demographic fairness analysis. Absence of an observed
demographic disparity must not be interpreted as evidence of fairness.

### Publication-selection bias

Published figures are intentionally selected to illustrate findings and often
use favorable slices, contrast, or annotations. Performance on this corpus is
likely an optimistic bound relative to routine clinical images.

## Failure-mode triage

| Failure flag | Records | Share |
|---|---:|---:|
| Multiple lesions or panels | 8,249 | 88.3% |
| Low IoU | 2,612 | 27.9% |
| Very small tumor | 141 | 1.5% |
| Low contrast | 81 | 0.9% |
| Empty mask | 1 | 0.01% |
| Any flag | 8,330 | 89.1% |

Categories overlap. Manual inspection reported that the dominant
multiple-lesion flag is usually caused by multi-panel figure composition.

## Reporting recommendations

Downstream studies should:

- report per-modality results as primary outcomes;
- state the bounding-box provenance;
- separate text-grounded and whole-image-fallback cases;
- report sample counts with all metrics;
- avoid inference from groups with fewer than 30 examples;
- use group-aware splitting at article or figure level;
- inspect caption and image duplication;
- disclose weak-label uncertainty;
- audit errors by layout, modality, and disease; and
- avoid clinical-deployment claims.

## Priority improvements

1. Decompose multi-panel figures before localization.
2. Replace keyword labels with a validated caption classifier.
3. Increase the proportion of valid text-grounded boxes.
4. Add source-license and provenance checks.
5. Expand under-represented modalities deliberately.
6. Establish article-level leakage-resistant splits.

