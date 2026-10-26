# A1 Backdoor for WiFi Pose Estimation

This repository follows the folder structure of [fiyud/wbackdoor](https://github.com/fiyud/wbackdoor). The main training entry point implements the locally modified **A1 micro-Doppler backdoor**. The upstream reference is commit `73871b02efa4291f8e476f845c26fe70a56cd471`.

```text
attack/             A1 trigger, pose payload, poisoning dataset
  adapted/          Six selected PiW3D CSI baseline adaptations
models/             HPE-Li and MetaFi++ victim models
data_utils/         Person-in-WiFi-3D feeder
eval/               Pose and dose-response metrics
configs/attack.yaml  Single A1 configuration
train_backdoor.py   Train and evaluate A1
sweep.py            Sweep A1 dose/poison settings
```

The six adapted method operators are WaNet, FTrojan, FIBA, BppAttack,
Smooth-LF-CSI, and TSBA. They are provided under `attack/adapted/` for
method inspection and for use by an experiment controller. The single
`configs/attack.yaml` and `train_backdoor.py` run A1 only; they do not train
these baseline victims. These are CSI/pose adaptations, not unchanged
reproductions of the source image/classification attacks.

Experiment results, datasets, and checkpoints are excluded.

## Run

1. Prepare the Person-in-WiFi-3D data. The feeder expects `csi_ap/<name>.npy` of shape `(3, 180, 20)`, formed by concatenating amplitude and phase along the second dimension.
2. Set `dataset_root` and `action_npy` in `configs/attack.yaml` to local paths. Verify its pivot and target joints against the dataset joint ordering.
3. Run `python train_backdoor.py --config configs/attack.yaml`.

The current YAML is the local A1 configuration. It uses right-hip pivot 7, targeting the right knee and ankle (joints 11 and 13). Paths in that file are examples from the local experiment and must be changed on another machine.
