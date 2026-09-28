# Hardware assessment

**Status:** P0 is qualified on the local GPU. The final activation experiment
is not hardware-qualified because it is scientifically blocked before a final
protocol exists.

## Measured P0 environment

| Field | Measured value |
| --- | --- |
| Backend | Native Windows CUDA; WSL was not used |
| GPU | NVIDIA GeForce RTX 3070 Ti Laptop GPU |
| Reported GPU memory | 8,589,410,304 B (8.000 GiB) |
| NVIDIA driver / CUDA UMD | 610.62 / 13.3 |
| PyTorch / CUDA build | 2.11.0+cu128 / 12.8 |
| Transformers / math-verify | 4.52.4 / 0.9.0 |
| CUDA available / BF16 supported | Yes / Yes |
| Python | CPython 3.12.3 |
| Model load reserved VRAM | 3,451,912,192 B (3.215 GiB) |
| Largest P0 generation reserved peak | 4,273,995,776 B (3.980 GiB) |
| P0 headroom at that peak | 4,315,414,528 B (4.019 GiB) |

## What the measurements support

| Workload | Assessment | Evidence |
| --- | --- | --- |
| Stochastic trajectory generation | Demonstrated | 120/120 base trajectories completed; 344,087 tokens at 13.855 aggregate tokens/s. |
| Forced-answer generation | Demonstrated | 464 available checkpoints completed; 207,347 tokens at 16.480 aggregate tokens/s. |
| P0 artifact storage | Demonstrated | Final local run directory: 11.436 MiB. |
| Checkpoint forward pass for a later study | Not qualified | P0 did not request hidden states or retain activation tensors; generation headroom is not an activation-memory measurement. |
| All-layer activation extraction | Not qualified | No final activation representation, number of units, serialization format, or protocol was frozen, and no all-layer extraction was tested. |
| Complete activation-probe experiment | Not appropriate to run | The scientific event-yield gate failed before a final experiment could be designed. This is a scientific block, not a P0 GPU failure. |

The combined P0 generation workload took 37,417.40 seconds (10.394 recorded
generation hours) for 551,434 generated tokens. That measurement is useful as
a P0 reference only. A final trajectory, forced-answer, activation-extraction,
total-GPU-hour, and activation-storage estimate would require a valid frozen
population and representation; neither exists after the scientific block, so
such a forecast would be fabricated.

## Conditional future hardware guidance

No hardware purchase or cloud run is justified for the blocked study. If a
separate, explicitly authorized protocol first establishes a viable target
population, use a small CUDA activation-feasibility run before committing to a
full campaign.

For that conditional future work, a 24 GB CUDA GPU is the practical planning
floor, not a P0-proven requirement: it offers substantially more room than the
current 8 GB device for one-at-a-time prefix forward passes and immediate CPU
offload. A cost-conscious first choice is a rented or already available RTX
3090/4090-class, RTX A5000, or A10-class 24 GB CUDA device. An A100-class
device is not justified by P0 measurements alone. NVIDIA documents 24 GB
memory for the [RTX 4090](https://www.nvidia.com/en-us/geforce/news/rtx-40-series-graphics-cards-announcements/)
and [RTX A5000](https://www.nvidia.com/en-us/products/workstations/rtx-a5000/).

Apple Silicon would not preserve the current CUDA execution backend. Apple
lists M2 Max configurations with up to 96 GB unified memory and M5 Pro
configurations with up to 64 GB, but PyTorch execution there uses the Metal
MPS backend rather than CUDA. That could be useful for a separately validated
port, but it cannot silently replace the frozen native-CUDA protocol or supply
comparable throughput/reproducibility evidence. See Apple’s
[M2 Max announcement](https://www.apple.com/newsroom/2023/01/apple-unveils-m2-pro-and-m2-max-next-generation-chips-for-next-level-workflows/),
[M5 Pro specifications](https://support.apple.com/en-ie/126319), and
[PyTorch-on-Metal guidance](https://developer.apple.com/metal/pytorch/).
