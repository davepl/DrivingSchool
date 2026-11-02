# Dave's Garage — GPU Driving School

262,144 independent cars learn steering and speed control on the Dell GB300.
The GPU runs both 2D vehicle simulation and PPO reinforcement learning. A local
browser dashboard replays 384 drivers at human speed and shows real training
throughput, crashes, miles, and held-out lap completion.

On the original Dell GB300, **max-load mode** used a 2,048-wide, 4.24-million-
parameter policy, 32-step rollouts, 131,072-sample minibatches, three PPO epochs,
and continuous training. New service installations use the smaller default profile. A one-minute measurement recorded **100% GB300
utilization in all 60 samples**, **1,037 W average power**, and a peak of 1,055 W.
The live viewer's average frame age was 94 ms. Those are measured whole-GPU
readings with Robotron still running, not a claim about advertised peak FLOPS.

## Watch on the Dell over RDP

1. Connect your RDP client to **dell**, using your existing login.
2. Open Firefox at **http://127.0.0.1:8780/**.
3. Click the fullscreen icon at the top right for the filming layout.
4. Select **Day one · total mayhem** to watch untrained drivers crash.
5. Select **best-road** or **best-barriers** to see what the policy learned.
6. Enable **Unseen test tracks** for courses excluded from gradient training.
7. Click **New class** to watch learning start over. Earlier classes are preserved
   in `runs/`; the current class's snapshots are available in the driver selector.

A **Driving School.desktop** launcher is installed in `/home/dave/Desktop`.
If your desktop does not show icons, open that folder in Files or use the URL.
GNOME may require right-clicking the launcher and choosing **Allow Launching**.
The Dell's existing GNOME RDP service listens on port 3389.

Service installation leaves autostart disabled and does not launch the program.
Start the service explicitly when ready. **New class** starts another run with fresh weights and preserves
the previous class. **Pause learning** pauses at an update
boundary while the live replay continues.

## Start and stop

```bash
cd /home/dave/source/repos/DrivingSchool
systemctl --user start driving-school
systemctl --user stop driving-school
journalctl --user -u driving-school -f
```

For a manually configured run, first stop the service, then:

```bash
./scripts/run.sh --max-load
```

`--max-load` selects the larger continuous profile. Explicit arguments override
its preset values. `--updates 0` trains continuously. `--envs` changes the real GPU population.
`--eager` disables compilation for debugging; `--device cpu` supports a small CPU
run. The scripts default to GPU 0, honoring `CUDA_VISIBLE_DEVICES` or an explicit
`DRIVING_GPU` override. On the original Dell, use `DRIVING_GPU=1` to select the
GB300. CUDA sees the selected device as `cuda:0`. Max-load mode runs replay inference and asynchronous
validation in separate processes exposing physical GPU 0, the RTX PRO 2000.
This separation avoids the Dell's CUDA initialization failure when both GPU
types are exposed in one process. `--viewer-device` and `--eval-device` accept
`external:0` for this arrangement. Replay uses BF16 inference; validation uses
the FP32 learned policy and the same fixed-seed tests.

For the original lighter profile, run `./scripts/run.sh` without `--max-load`.
It uses 65,536 cars, a 512-wide policy, and 300 updates, then selects its best
driver. The smaller profile is useful for quick learning demonstrations;
the max-load profile is intended to occupy the GB300 continuously.

On another computer, set `DRIVING_PYTHON` as needed, then run
`scripts/setup.sh`. It creates `.venv` and installs dependencies without starting
anything. Set `DRIVING_TORCH_INDEX` if a specific official PyTorch wheel index is
needed for the target driver or a CPU installation.
The checked-in `requirements.lock.txt` records the Dell's exact runtime packages.
On this Dell the isolated `.venv` uses CPython 3.14.8, PyTorch 2.13.0+cu130, and
NumPy 2.5.2. It does not alter the existing Robotron runtime.

For viewing from another machine without RDP:

```bash
ssh -N -L 8780:127.0.0.1:8780 dell
```

Open http://127.0.0.1:8780/ locally. The dashboard binds to loopback by default.
Controls reject cross-origin requests and non-loopback clients. RDP configuration
and credentials are managed by the existing Dell setup.

## What is real

- Every population member has its own position, orientation, speed, steering,
  episode progress, and observation tensor on the GPU.
- Steering and desired speed come from a Gaussian neural policy, optimized with
  clipped PPO, generalized advantage estimation, a value loss, and entropy.
- Vehicle dynamics use a kinematic bicycle model with acceleration, braking,
  steering response, and a fixed 0.1-second simulation step.
- Observations contain road offset, heading, speed, bends ahead, and nearby
  barriers. They contain no track ID, target steering command, or expert action.
- Rewards depend on forward distance, staying on the road, completing laps,
  and avoiding crashes. Off-road, wrong-way, and barrier crashes reset that car.
- Training uses six procedural closed circuits. Two different circuits are used
  for validation with 128 fixed-seed starting conditions. A success requires one
  complete lap without failure. Those courses receive no gradient updates;
  their scores are used for curriculum advancement and best-checkpoint selection.
- Barriers are introduced once road-only validation reaches 80% after at least
  20 updates. Road and barrier checkpoints remain separate.
- The browser shows a separate human-speed simulation of the **actual neural policy**,
  rather than the entire population rendered in real time. Replay physics run
  on the CPU, and inference runs on the isolated RTX GPU in max-load mode. It is labelled as a live policy
  replay. Cars overlap because their training worlds are independent; they do
  not collide with each other.

This is a lightweight driving game and learning demo. Its huge population makes
parallel learning visible. It is not a traffic simulator or a physical-car model.

## Measurements on this Dell

Current max-load measurements are saved in
`benchmarks/dell-2026-10-14/max-load-2048.json` and its raw `.jsonl` trace.
Updates 38–58 progressed during the 60-second measurement, including background
validation. Every utilization sample was 100%; peak temperature was 62°C.
The larger policy processed about 3 million car-steps per second. That rate
has more computation per car than the original profile below, so the two rates
are not a speed comparison of identical workloads. Validation runs on the RTX
without stopping the GB300's training, and policy copies publish every five updates.

To measure load again while the service is running:

```bash
python3 scripts/measure-load.py --seconds 60
```

An identical compiled PPO workload with 65,536 environments, a 512-wide network,
32-step rollouts, three optimization epochs, and 65,536-sample minibatches gave:

| GPU | Median car-steps / second | Median update time |
| --- | ---: | ---: |
| GB300 | 9,848,427 | 0.213 s |
| RTX PRO 2000 Blackwell | 1,177,131 | 1.782 s |

That is approximately **8.37×** the throughput on the GB300. Each measurement
uses updates 3–20, excluding two warmup updates. Timing includes rollout and PPO
optimization, with CUDA synchronization; it excludes validation, checkpoint writes,
and display. FP32 tensors use PyTorch's high matmul precision setting, permitting
TF32. Existing jobs were left running, including Robotron on the GB300. This is
a workload comparison against the Dell's smaller graphics card, not a claim
about every GPU or the machine's advertised peak FLOPS.

The live 65,536-car run measured roughly 9 million car-steps/s during training
and roughly 4 million/s averaged across startup and periodic tests. Its rookie
completed 0% of validation laps; road-only training reached 100%, and the best
barrier policy reached 95.3%. Later policy updates can regress, so the best
checkpoint is saved and selected when a class finishes.

Raw benchmark results are in `benchmarks/dell-2026-10-14/`. To repeat:

```bash
PYTHON_GIL=1 .venv/bin/python scripts/benchmark.py --gpu 1 --envs 65536 --width 512 --batch 65536
PYTHON_GIL=1 .venv/bin/python scripts/benchmark.py --gpu 0 --envs 65536 --width 512 --batch 65536
```

The dashboard's main rate is the most recently measured training update. The
smaller rate includes startup and tests. Miles sum actual forward travel across
the population; simulated hours sum world-time across all cars. The hardware
panel's power and utilization include other GPU processes. Training memory is
this process's peak tensor allocation, not the entire GPU's memory use.

## Code and validation

`drivingschool/env.py` implements GPU tensor physics and sensors;
`policy.py` defines the actor and critic; `train.py` implements PPO and testing;
`live.py` serves the dashboard and runs the human-speed policy replay;
`worker.py` isolates RTX replay and validation from the GB300 trainer. The viewer
uses local Canvas graphics and system fonts, with no cloud APIs or external assets.

```bash
PYTHON_GIL=1 .venv/bin/python -m unittest discover -s tests -v
```

Tests cover finite simulation, reset behavior, barrier collisions, lap completion,
terminal advantage calculation, real policy optimization, and repeatable validation.
They also verify that asynchronous checkpoints preserve the exact policy and
update that were tested, and that inference copies load every model parameter.
The compiled GPU training path and browser controls are also checked on the Dell.

Each class writes `metrics.jsonl`, `rookie.pt`, `latest.pt`, `best-road.pt`,
`best-barriers.pt`, and numbered lesson checkpoints under `runs/`. JSON sidecars
record scores and settings. A service restart starts a new class and preserves
earlier checkpoints. A checkpoint contains policy weights and metadata, not the
optimizer or live simulation state; it is intended for reviewing a learned driver.

## A good on-camera sequence

Start with rookies at 4× speed: crashes everywhere. Show the population counter,
then start a new class and watch its first test go from 0% toward full laps.
Switch to the best road driver and the unseen tracks. Introduce the barrier lesson
and its saved successful driver. Finish with the measured 8.37× GPU comparison.
