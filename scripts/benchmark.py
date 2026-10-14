"""Run the same PPO workload on a selected CUDA device. No rendering included."""
import argparse
import json
import os
import statistics
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=1)
parser.add_argument("--envs", type=int, default=16384)
parser.add_argument("--width", type=int, default=256)
parser.add_argument("--batch", type=int, default=32768)
parser.add_argument("--updates", type=int, default=20)
parser.add_argument("--eager", action="store_true")
opts = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(opts.gpu)
os.environ["PYTHON_GIL"] = "1"
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import torch
import threading
from drivingschool.train import Trainer
from drivingschool.live import Bridge

torch.set_num_threads(2)
torch.set_float32_matmul_precision("high")
torch.cuda.set_device(0)
args = argparse.Namespace(device="cuda:0", envs=opts.envs, width=opts.width,
    updates=opts.updates, horizon=32, batch=opts.batch, epochs=3, eval_every=100000,
    eager=opts.eager, seed=42)
run_dir = Path("runs") / f"benchmark-gpu{opts.gpu}-n{opts.envs}-w{opts.width}-b{opts.batch}"
run_dir.mkdir(parents=True,exist_ok=True)
(run_dir/"metrics.jsonl").write_text("")
Trainer(args, Bridge(), run_dir, threading.Event()).run()
rows = [json.loads(line) for line in (run_dir/"metrics.jsonl").read_text().splitlines()]
rows = rows[-(opts.updates-2):]
result = {"gpu":torch.cuda.get_device_name(0), "population":opts.envs, "width":opts.width,
    "horizon":32, "epochs":3, "batch":opts.batch, "precision":"FP32 with high matmul precision (TF32 permitted)",
    "torch":torch.__version__, "median_car_steps_per_second":statistics.median(r["steps_per_second"] for r in rows),
    "median_update_seconds":statistics.median(r["update_seconds"] for r in rows),
    "measured_updates":len(rows), "warmup_updates_excluded":2,
    "notes":"Includes rollout + PPO optimization. Tests, checkpoint writes and rendering excluded. Other GPU jobs were not stopped."}
output = run_dir/"benchmark.json"
output.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
