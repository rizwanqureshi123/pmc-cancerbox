# Process all relevant PMC images from every labeled JSON in the repo root.
# Requires CUDA-enabled PyTorch.

param(
    [switch]$RequireBbox
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$jsonFiles = @(
    "..\labeled-dataset-1783958094047.json",
    "..\labeled-dataset1-1778366141328.json",
    "..\labeled_dataset2_00001_unlabeled.json",
    "..\labeled-dataset4-1779744834686_ (1).json",
    "..\labeleddataset3-1779033606801 (1).json",
    "..\labeled-dataset-1783650371260.json"
)

$argsList = @(
    "infer_pmc_labels.py",
    "--json"
) + $jsonFiles + @(
    "--checkpoint", "checkpoints/mcp_medsam.pth/mcp_best.pth",
    "--output-dir", "pmc_outputs",
    "--device", "cuda:0",
    "--save-overlay"
)

if (-not $RequireBbox) {
    $argsList += "--allow-full-image-bbox"
}

python @argsList
