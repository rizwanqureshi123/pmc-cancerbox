<<<<<<< HEAD
# MCP-MedSAM

Pytorch Implementation of the paper:
"[MCP-MedSAM: A Powerful Lightweight Medical Segment Anything Model Trained with a Single GPU in Just One Day](https://arxiv.org/abs/2412.05888)"

![MCP-MedSAM Architecture](docs/MCP-MedSAM.png)

## 📄 Overview

This work proposes a lightweight variant of MedSAM by integrating:

- A **pre-trained Tiny ViT** as the vision backbone  
- Two novel prompt types:  
  - **Modality Prompt**  
  - **Content Prompt**  
- A **modified mask decoder** adapted to these prompts  

To further improve performance across imaging modalities, we introduce a **modality-aware data sampling strategy** that ensures better balance and generalization.

With these enhancements, our model achieves strong multi-modality segmentation performance, and can be trained in approximately **1 day on a single A100 (40GB)** GPU.

<!-- 
We are currently releasing the inference code along with the model weight. You can download from [here](https://drive.google.com/drive/folders/1NW4aSNhk-dtiK-dicTAUp0g0eR2fryNi?usp=sharing).

The training code has been released and you can train your . -->

## Requirements

* Python==3.10.14
* torch==2.0.0
* torchvision==0.15.0
* transformers==4.49.0

## Training and Inference

Training and inference can be done by running train.py and infer.py. To note, there is a 'case_data.json' file in the custom dataset class, which is used for speeding up the reading of data, you can replace ? with your local data path. Additionally, we also release the weights of tiny ViT and the whole MCP-MedSAM for inference, which can be downloaded from [here](https://drive.google.com/drive/folders/1NW4aSNhk-dtiK-dicTAUp0g0eR2fryNi?usp=sharing). Furthermore, MCP-MedSAM has also been uploaded to the [Hugging Face](https://huggingface.co/Leo-Lyu/MCP-MedSAM), including pre-trained weights as well.

## PMC Labeled JSON Workflow

Two scripts bridge PMC relevance-labeled JSON files to MCP-MedSAM inference:

| Script | Purpose |
|--------|---------|
| `pmc_labeled_dataset.py` | Filter `relevant` images, infer modality from captions, extract bboxes from `regions` |
| `infer_pmc_labels.py` | Run MCP-MedSAM segmentation on the filtered scans |

### 1. Export a manifest (preview relevant scans)

```bash
python pmc_labeled_dataset.py \
  --json ../labeled-dataset-1783958094047.json \
  --output pmc_relevant_manifest.json
```

Optional: point `--image-dir` at a folder of downloaded PMC images to resolve local paths.

### 2. Run inference

Download the MCP-MedSAM checkpoint from the link above, then:

```bash
python infer_pmc_labels.py \
  --json ../labeled-dataset-1783958094047.json \
  --checkpoint checkpoints/mcp_medsam.pth/mcp_best.pth \
  --output-dir pmc_outputs \
  --allow-full-image-bbox \
  --save-overlay \
  --limit 10 \
  --device cpu
```

On PowerShell, use a single line (no `^` continuations):

```powershell
python infer_pmc_labels.py --json ../labeled-dataset-1783958094047.json --checkpoint checkpoints/mcp_medsam.pth/mcp_best.pth --output-dir pmc_outputs --allow-full-image-bbox --save-overlay --limit 10 --device cpu
```

**Flags:**
- `--allow-full-image-bbox` — use a centered full-image bbox when `regions`/`bbox` are missing (needed for current JSON files, which have relevance labels only)
- `--image-dir` — local PMC image folder (otherwise images are downloaded from URL into `--cache-dir`)
- `--limit` — cap processed scans for testing

**Outputs:**
- `output-dir/masks/*.npz` — mask, bbox, IoU, modality, caption
- `output-dir/overlays/*.png` — side-by-side visualizations (with `--save-overlay`)

### JSON format supported

Grouped by article:

```json
[
  {
    "pmcid": "138673",
    "images": [
      {
        "url": "https://cdn.ncbi.nlm.nih.gov/pmc/blobs/.../bcr266-1.jpg",
        "caption": "A T1-weighted postcontrast image...",
        "label": "relevant"
      }
    ]
  }
]
```

Flat per-image entries are also supported. Optional spatial annotations:

```json
{
  "regions": [{ "x": 120, "y": 80, "width": 200, "height": 150 }],
  "bbox": [120, 80, 320, 230]
}
```

When regions are present, bboxes are extracted automatically. Otherwise use `--allow-full-image-bbox` for exploratory segmentation.

### 3. Full batch on all JSON files (CUDA)

First install CUDA-enabled PyTorch (the default pip install is often CPU-only):

```powershell
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available())"
```

Then run all 6 labeled JSON files (~9,345 relevant images):

```powershell
.\run_all_pmc_inference.ps1
```

After you add bboxes to JSON, rerun with real boxes:

```powershell
.\run_all_pmc_inference.ps1 -RequireBbox
```

### 4. Add bounding boxes to JSON (recommended before final run)

Your JSON files currently have relevance labels only. Use the interactive annotator:

```powershell
python annotate_pmc_bboxes.py --json ../labeled-dataset-1783958094047.json --limit 50
```

Controls: drag to draw box, `s` save, `k` skip, `b` back, `q` quit.

Each saved box is written into the source JSON as:

```json
{
  "bbox": [120, 80, 320, 230],
  "regions": [{ "x": 120, "y": 80, "width": 200, "height": 150 }]
}
```

A `.bak` backup is created the first time each JSON file is modified.

**Practical workflow**

1. Annotate a small batch first (e.g. 50–100 images) and verify quality.
2. Run inference with `-RequireBbox` on that subset.
3. Scale annotation, then run `.\run_all_pmc_inference.ps1 -RequireBbox` for the full dataset.

For ~9,345 images, full manual annotation is a large effort. Consider annotating only the highest-value subset, or using `--allow-full-image-bbox` for an initial pass and refining bboxes later from the overlays.

## Citation

```bash
@article{lyu2024mcp,
  title={MCP-MedSAM: A Powerful Lightweight Medical Segment Anything Model Trained with a Single GPU in Just One Day},
  author={Lyu, Donghang and Gao, Ruochen and Staring, Marius},
  journal={arXiv preprint arXiv:2412.05888},
  year={2024}
}
```
=======
<p align="center">
  <img src="assets/repository-banner.svg" alt="PMC-CancerBox" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-publication%20proof-6f3cc3" alt="Publication proof">
  <img src="https://img.shields.io/badge/images%20and%20masks-9%2C345-3478d4" alt="9,345 image-mask pairs">
  <img src="https://img.shields.io/badge/diseases-breast%20cancer%20%7C%20brain%20tumor-d65374" alt="Breast cancer and brain tumor">
  <a href="LICENSE"><img src="https://img.shields.io/badge/repository%20documentation-CC%20BY%204.0-4a9b68" alt="CC BY 4.0"></a>
</p>

# PMC-CancerBox

**A Multi-Modal Breast Cancer and Brain Tumor Segmentation Dataset with a
Responsible-AI Representation Audit**

PMC-CancerBox is a scientific-literature-derived dataset and evaluation
framework for prompt-guided medical image segmentation. It contains 9,345
image-mask pairs assembled from open-access PubMed Central figures related to
breast cancer and brain tumors.

> The central finding is that localization—not mask refinement—is the main
> bottleneck: text-grounded prompts substantially outperform whole-image
> fallback prompts.

## Repository status

This repository is currently a **documentation and student-development
workspace**.

- The paper is at publication-proof stage.
- The final DOI and complete author metadata are not yet available in the
  supplied proof.
- Dataset access details will be added after the release location is finalized.
- The [`code/`](code/) directory intentionally contains **no implementation**.
  It is reserved for supervised student contributions.
- The publisher proof PDF is not distributed here.

## Study at a glance

| Item | Description |
|---|---|
| Source | PubMed Central Open Access scientific articles |
| Records | 9,345 image-mask pairs from 9,319 figure captions |
| Diseases | Breast cancer and brain tumor |
| Main modalities | MR, CT, ultrasound, mammography, microscopy, PET, X-ray |
| Annotation | Bounding-box localization followed by prompt-guided segmentation |
| Segmentation backbone | MCP-MedSAM |
| Training | No model retraining in the reported pipeline |
| Primary metric | Intersection over Union (IoU) |
| Responsible-AI component | Modality, disease, imaging-family, and failure-mode audit |

## Pipeline

<p align="center">
  <img src="assets/pipeline-overview.svg" alt="PMC-CancerBox pipeline overview" width="100%">
</p>

The reported workflow:

1. Harvests image URLs and metadata from PMC Open Access articles.
2. Extracts and stores figures with associated captions.
3. Filters figures for relevance to breast cancer and brain tumors.
4. Generates or records lesion bounding boxes.
5. Uses boxes as prompts for MCP-MedSAM.
6. Collects segmentation masks and calculates IoU.
7. Audits representation, confounding, and pipeline failure modes.

## Key results

### Segmentation

| Evaluation subset | Records | Mean IoU | Median IoU |
|---|---:|---:|---:|
| Text-grounded box available | 3,381 | **0.6332** | **0.6379** |
| Whole-image fallback | 5,798 | 0.4943 | 0.4498 |
| Unknown box provenance | 166 | 0.5640 | 0.5519 |
| **All records** | **9,345** | **0.5458** | **0.5150** |

Text-grounded boxes improve mean IoU by 0.139 over whole-image fallback,
indicating that better localization is the highest-value direction for
improving the pipeline.

### Representation audit

| Finding | Result |
|---|---:|
| MR and CT combined | 84.9% |
| Radiological images | 96.8% |
| Captions containing at least one medical term | 91.95% |
| Records in the heuristic “other” disease category | 48.0% |
| Records flagged as multiple lesions/panels | 88.3% |
| Articles published in 2020 or later | 83.4% of dated articles |

The apparent “multiple lesions” failure is largely a multi-panel publication
figure artifact rather than evidence of multifocal disease.

## Modality profile

| Modality | Records | Corpus share | Mean IoU |
|---|---:|---:|---:|
| MR | 4,077 | 43.6% | 0.4634 |
| CT | 3,861 | 41.3% | 0.5885 |
| Ultrasound | 529 | 5.7% | 0.7233 |
| Mammography | 421 | 4.5% | 0.5752 |
| Microscopy | 298 | 3.2% | 0.7232 |
| PET | 85 | 0.9% | 0.6311 |
| X-ray | 66 | 0.7% | 0.5800 |

Results for modalities with fewer than 30 records should not be interpreted as
reliable estimates. See [docs/DATA_CARD.md](docs/DATA_CARD.md) for the full
dataset description.

## Responsible use

PMC-CancerBox is a research dataset, not a clinical product.

- Patient demographics are absent from PMC figure metadata; demographic
  fairness analysis is therefore out of scope for the current release.
- Published figures are selected for illustrative clarity and do not represent
  consecutive clinical cohorts.
- Disease labels derived from caption keywords are weak labels, not clinical
  ground truth.
- Modality and disease are confounded.
- Aggregate performance must not be interpreted as evidence of deployment
  readiness.

Read the detailed [responsible-AI audit](docs/RESPONSIBLE_AI_AUDIT.md).

## Student implementation area

The [`code/`](code/) directory is intentionally empty of source code. Students
can implement independent modules for:

- PMC acquisition and metadata normalization;
- figure-type and relevance classification;
- multi-panel figure decomposition;
- caption-based disease classification;
- text-grounded bounding-box generation;
- MCP-MedSAM, Swift-MedSAM, and MedSAM evaluation;
- reproducible metrics and error analysis; and
- automated representation audits.

Suggested projects and acceptance criteria are provided in
[docs/STUDENT_PROJECTS.md](docs/STUDENT_PROJECTS.md).

## Repository structure

```text
pmc-cancerbox/
├── README.md
├── CITATION.cff
├── LICENSE
├── CONTRIBUTING.md
├── assets/
│   ├── repository-banner.svg
│   └── pipeline-overview.svg
├── code/
│   └── README.md
├── data/
│   └── README.md
├── docs/
│   ├── DATA_CARD.md
│   ├── RESPONSIBLE_AI_AUDIT.md
│   ├── STUDENT_PROJECTS.md
│   └── REFERENCES.md
└── .github/
    ├── ISSUE_TEMPLATE/
    └── pull_request_template.md
```

## Citation

The final bibliographic record is pending publication. A provisional citation
file is provided in [CITATION.cff](CITATION.cff) and should be updated when the
DOI and complete author list are confirmed.

## Copyright

The publication proof, paper figures, and publisher layout are not included.
Original repository documentation and graphics are licensed under
[CC BY 4.0](LICENSE). External papers, data, models, and code retain their own
terms.

>>>>>>> origin/main
