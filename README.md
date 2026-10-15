# PromptHive

PromptHive is an adaptive point-prompt optimization framework for SAM image segmentation. It combines feature relationships and spatial relationships, and uses DQN to adjust positive and negative prompt points to improve segmentation quality.

## Features

- Supports five-dimensional statistical states and 22-dimensional graph states;
- Supports neural network policy and policy table diagnostic modes;
- Supports FSS-1000, VOC2012, ISIC2018, and Kvasir;
- Outputs per-image results, mean Dice, and HD95;
- Provides a single-dataset entry point and a batch script for all four datasets.

## Installation

Python 3.10 or later is recommended. Install PyTorch according to your local CUDA environment:

python -m pip install -r requirements.txt

## Weights

Download the PromptHive weight package from the project's Release page and extract it into the project root directory. The directory structure should be:
```lua
weights/
|-- best-strict.pt
|-- best-graph.pt
`-- best-policy-table.pkl

```

| **Mode**      | **Weights**                      | **Purpose**                  |
| ----------- | ------------------------------- | ---------------------------- |
| `pt-strict` | `weights/best-strict.pt`        | Five-dimensional statistical state policy |
| `pt-graph`  | `weights/best-graph.pt`         | 22-dimensional graph state policy |
| `pkl`       | `weights/best-policy-table.pkl` | Policy table diagnostics      |

## Single-Dataset Evaluation

python run_eval.py \   --mode pt-strict \   --bundle-dir /path/to/features \   --image-dir /path/to/images \   --mask-dir /path/to/masks \   --sam-checkpoint /path/to/sam_vit_l_0b3195.pth \   --output-dir results/pt-strict \   --device cuda:0

`--limit 0` evaluates all samples. The output directory contains `summary.json` and `per_image.csv`.

## Four-Dataset Evaluation

MODE=pt-strict LIMIT=1 \ DATA_ROOT=/path/to/datasets \ SAM_CHECKPOINT=/path/to/sam_vit_l_0b3195.pth \ bash scripts/run_all_four.sh

Set `LIMIT` to `0` to run the full evaluation. Change `MODE` to `pt-graph` or `pkl` to switch policies.

## Metrics

- `mDSC_percent`: mean Dice, reported as a percentage;
- `mHD`: HD95 calculated at an evaluation scale of 560 x 560;
- `pretrained_qtable_key_hits`: number of key hits in the policy table.

## Testing

python -m unittest discover -s prompthive/tests -v
