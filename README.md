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
