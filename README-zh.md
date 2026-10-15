# PromptHive

PromptHive 是一个面向 SAM 图像分割的自适应点提示优化框架。它结合特征
关系与空间关系，通过 DQN 调整正、负提示点，以提升分割质量。

## 功能

- 支持五维统计状态和 22 维图状态；
- 支持神经网络策略与策略表诊断模式；
- 支持 FSS-1000、VOC2012、ISIC2018 和 Kvasir；
- 输出逐图结果、平均 Dice 和 HD95；
- 提供单数据集入口和四数据集批量脚本。

## 安装

建议使用 Python 3.10 或更高版本，并根据本机 CUDA 环境安装 PyTorch：

```bash
python -m pip install -r requirements.txt
```

## 权重

从项目 Release 页面下载 PromptHive 权重包并解压到项目根目录。目录结构应为：

```text
weights/
|-- best-strict.pt
|-- best-graph.pt
`-- best-policy-table.pkl
```

| 模式 | 权重 | 用途 |
| --- | --- | --- |
| `pt-strict` | `weights/best-strict.pt` | 五维统计状态策略 |
| `pt-graph` | `weights/best-graph.pt` | 22 维图状态策略 |
| `pkl` | `weights/best-policy-table.pkl` | 策略表诊断 |

## 单数据集评估

```bash
python run_eval.py \
  --mode pt-strict \
  --bundle-dir /path/to/features \
  --image-dir /path/to/images \
  --mask-dir /path/to/masks \
  --sam-checkpoint /path/to/sam_vit_l_0b3195.pth \
  --output-dir results/pt-strict \
  --device cuda:0
```

`--limit 0` 表示评估全部样本。输出目录包含 `summary.json` 和
`per_image.csv`。

## 四数据集评估

```bash
MODE=pt-strict LIMIT=1 \
DATA_ROOT=/path/to/datasets \
SAM_CHECKPOINT=/path/to/sam_vit_l_0b3195.pth \
bash scripts/run_all_four.sh
```

将 `LIMIT` 设为 `0` 可运行全量评估；将 `MODE` 改为 `pt-graph` 或
`pkl` 可切换策略。

## 指标

- `mDSC_percent`：平均 Dice，百分数；
- `mHD`：在 560 x 560 评估尺度计算的 HD95；
- `pretrained_qtable_key_hits`：策略表中的键命中次数。

## 测试

```bash
python -m unittest discover -s prompthive/tests -v
```
