# A1 backdoor for WiFi pose estimation

The top-level folder layout follows [fiyud/wbackdoor](https://github.com/fiyud/wbackdoor) (upstream commit `73871b02efa4291f8e476f845c26fe70a56cd471`). The **A1 implementation here follows the later MM-Fi and Person-in-WiFi-3D experiments**, rather than the older complex CSI trigger in that upstream repository.

```text
attack/              A1 trigger, pose targets, poisoning, six adapted baseline operators
models/              HPE-Li and MetaFi++ PiW victim models
data_utils/           Person-in-WiFi-3D feeder
eval/                 Pose metrics
configs/attack.yaml  PiW HPE-Li seed-42 A1 configuration
train_backdoor.py     PiW A1 training and evaluation
```

## A1 method

`attack/trigger.py` constructs the action-derived complex micro-Doppler pattern. Its real projection is centered and scaled to unit RMS (`p0`). `attack/multicarrier.py` constructs the seeded separable DCT-II carrier, removes its projection onto `p0`, and scales it to unit RMS (`p1`). The fixed A1 pattern is the unit-RMS version of `(p0 + p1)/sqrt(2)`. At dose `d`, normalized CSI amplitude `x` becomes `clip(x * clip(1 + d * 0.185 * pattern, 0, infinity), 0, 1)`.

- **MM-Fi:** The operator applies to normalized amplitude `(3,114,10)`. The target rotates joints 2 and 3 around pivot 1 and the fixed Z axis by up to 40 degrees. See `attack/mmfi_a1.py`. The MM-Fi model training and evaluation controller used for reported results remains in `backdoor-analog/ATKBackd`; this repository contains its A1 method operator, not a standalone reproduction of the MM-Fi tables.
- **Person-in-WiFi-3D:** The same numerical operator applies separately to three normalized `(3,30,20)` amplitude groups; phase rows are unchanged. The target rotates right knee/ankle (11,13) around right hip (7) using the per-pose axis `normalize(GT[5]-GT[7])`, up to 40 degrees. See `attack/piw_a1.py`. The PiW seed-42 configuration uses 10% uniform poisoning, paired doses in `[0.2,1]`, AdamW `lr=0.001`, batch 32, and 200 epochs.

The six source-inspired baseline operators (WaNet, FTrojan, FIBA, BppAttack, Smooth-LF-CSI, TSBA) remain in `attack/adapted/`. `train_backdoor.py` runs **A1 only**.

## Run PiW A1

Set `dataset_root` and `action_npy` in `configs/attack.yaml`, then run:

```bash
python train_backdoor.py --config configs/attack.yaml --ckpt-dir runs/piw_a1
```

The feeder expects each PiW CSI input to have shape `(3,180,20)` with 90 amplitude rows followed by 90 phase rows. Training normalizes before A1 injection, as in the seed-42 adaptation. The action file's SHA-256 is bound into the checkpoint configuration.

This is a **standalone PiW A1 runner**. The full recorded v3 experiment used a separate Clean+A1 controller, complete input checks, deferred TEST evaluation, and its own result format. Running this entry point alone does not reproduce that complete comparison or assert the previously reported numerical results. Results, datasets and checkpoints are excluded from Git.
