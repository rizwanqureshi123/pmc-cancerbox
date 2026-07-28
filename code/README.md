# Student implementation area

This directory intentionally contains **no source code**.

It is reserved for student implementations of the PMC-CancerBox pipeline.
Before contributing code, read:

- [Student projects](../docs/STUDENT_PROJECTS.md)
- [Dataset card](../docs/DATA_CARD.md)
- [Responsible-AI audit](../docs/RESPONSIBLE_AI_AUDIT.md)
- [Contribution guidelines](../CONTRIBUTING.md)

## Expected future organization

Students may propose modules such as:

```text
code/
├── acquisition/
├── curation/
├── panel_decomposition/
├── grounding/
├── segmentation/
├── evaluation/
└── audit/
```

Do not create these directories until a project has been assigned and its
scope has been approved.

## Minimum expectations

Every implementation should eventually include:

- a clear command-line entry point;
- configuration separated from source code;
- deterministic seeds where applicable;
- provenance-preserving input and output schemas;
- unit tests for transformations and metrics;
- small synthetic fixtures rather than patient or restricted data;
- documented failure behavior; and
- an explicit license compatible with all dependencies.

The repository is not a clinical device and student code must not be described
as suitable for clinical use.

