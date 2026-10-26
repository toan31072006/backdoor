# Analog (Dose-Response) Backdoor against WiFi Pose Estimation

This repository keeps the folder structure of [fiyud/wbackdoor](https://github.com/fiyud/wbackdoor) and includes local changes to its attack methods, models, data feeder, evaluation, and experiment configs. The upstream reference is commit `73871b02efa4291f8e476f845c26fe70a56cd471`.

It contains source code only. Datasets, action arrays, checkpoints, caches, and experiment results are excluded.

## Layout

```text
attack/             Trigger, payload, poisoning pipeline, TSBA adaptation
models/             HPE-Li and MetaFi++ victims
data_utils/         Person-in-WiFi-3D feeder
eval/               Pose and CSI-distortion metrics
configs/            Experiment configurations
train_backdoor.py   Train and evaluate a single configuration
sweep.py            Run a configuration sweep
tests/              Unit and regression tests
run_colab.ipynb     Colab workflow
run_piw3d.ipynb     Person-in-WiFi-3D workflow
```

## Running

1. Prepare Person-in-WiFi-3D data using the dataset's official preprocessing. The feeder expects `csi_ap/<name>.npy` with shape `(3, 180, 20)`, made by concatenating amplitude and phase along the second dimension.
2. Confirm `pivot` and `target_joints` against the dataset's joint ordering. The supplied Person-in-WiFi-3D configuration uses right hip pivot 7 and right knee/ankle targets 11 and 13.
3. Set `dataset_root` and `action_npy` in the selected YAML file to local paths. Example paths in `configs/` are from the original experiments and are not portable.
4. Run `python train_backdoor.py --config configs/attack.yaml`.

The `configs/tsba.yaml` variants use a sample-specific digital CSI trigger while retaining the HPE-Li victim, target pose, and pose-regression evaluation. They are **TSBA-adapted** experiments, not exact reproductions of the original classification attack.

## Scope

The micro-Doppler and TSBA experiments operate on stored CSI arrays. Results require training and evaluation with the specified real dataset and split. ImageNet weights are needed when a MetaFi++ configuration enables pretrained initialization.
