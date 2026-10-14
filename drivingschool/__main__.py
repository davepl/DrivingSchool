import argparse
import datetime
import signal
import threading
import traceback
import subprocess
import os
from pathlib import Path
import torch
from .live import Bridge, spectator, hardware_monitor, serve
from .train import Trainer


def main():
    parser = argparse.ArgumentParser(description="Dave's Garage GPU Driving School")
    parser.add_argument("--max-load", action="store_true", help="Large, continuous GPU training preset")
    parser.add_argument("--envs", type=int, default=65536)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--batch", type=int, default=65536)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--updates", type=int, default=300, help="0 means continuous")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--publish-every", type=int, default=None, help="Publish policy and checkpoint interval")
    parser.add_argument("--viewer-device",default=None)
    parser.add_argument("--eval-device",default=None,help="Run validation asynchronously on this device")
    args = parser.parse_args()
    if args.max_load:
        # Defaults can still be overridden explicitly for tuning or other GPUs.
        import sys
        preset = {"envs":262144, "width":2048, "batch":131072, "epochs":3,
                  "eval_every":20, "updates":0, "publish_every":5}
        for name,value in preset.items():
            flag="--"+name.replace("_","-")
            if not any(a==flag or a.startswith(flag+"=") for a in sys.argv[1:]):
                setattr(args,name,value)
    if args.publish_every is None:
        args.publish_every=1
    if min(args.envs,args.horizon,args.width,args.batch,args.epochs,args.eval_every,args.publish_every) <= 0:
        parser.error("Population and training sizes must be positive")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.set_float32_matmul_precision("high")
    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.device)
    if args.max_load and args.device.startswith("cuda"):
        # Mixed GB300/RTX visibility fails CUDA initialization on this Dell.
        # Separate processes each expose exactly one physical device.
        if args.viewer_device is None:
            args.viewer_device="external:0"
        if args.eval_device is None:
            args.eval_device="external:0"
    if args.viewer_device is None:
        args.viewer_device=args.device
    bridge = Bridge()
    stop = threading.Event()
    root = Path(__file__).resolve().parent.parent
    # Old classes remain reviewable after a service restart.
    import json
    for metadata in sorted(Path(args.runs).glob("class-*/*.json")):
        checkpoint = metadata.with_suffix(".pt")
        if checkpoint.is_file():
            try:
                bridge.register_checkpoint(checkpoint, json.loads(metadata.read_text()))
            except (ValueError, OSError):
                pass
    server = serve(bridge, root/"web", args.host, args.port)
    replay_worker=None
    if args.viewer_device.startswith("external:"):
        import sys
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES=args.viewer_device.split(":",1)[1])
        replay_worker=subprocess.Popen([sys.executable,"-m","drivingschool.worker","replay","--port",str(args.port)],env=environment)
    else:
        threading.Thread(target=spectator,args=(bridge,stop,384,args.viewer_device),daemon=True).start()
    threading.Thread(target=hardware_monitor,args=(bridge,stop),daemon=True).start()

    def supervise():
        while not stop.is_set():
            class_stop = threading.Event()
            run_dir = Path(args.runs)/datetime.datetime.now(datetime.timezone.utc).strftime("class-%Y%m%d-%H%M%S")
            with bridge.lock:
                bridge.metrics = {"status":"Starting new class", "steps":0, "run":run_dir.name}
                bridge.policy_state = None
                bridge.frame = {}
                bridge.history = []
                bridge.view["driver"] = "latest"
                bridge.paused = False
            def learn():
                try:
                    Trainer(args,bridge,run_dir,class_stop).run()
                except Exception as error:
                    traceback.print_exc()
                    bridge.update_metrics(status="Training error", error=str(error))
            learner = threading.Thread(target=learn, daemon=True)
            learner.start()
            while not stop.is_set() and not bridge.new_class.wait(.2):
                pass
            class_stop.set()
            bridge.update_metrics(status="Saving class before restart")
            learner.join()
            bridge.new_class.clear()
    supervisor = threading.Thread(target=supervise,daemon=True)
    supervisor.start()
    def shutdown(*_):
        stop.set()
        threading.Thread(target=server.shutdown,daemon=True).start()
    signal.signal(signal.SIGTERM,shutdown)
    signal.signal(signal.SIGINT,shutdown)
    print(f"Driving School: http://{args.host}:{args.port} | {args.envs:,} cars on {args.device}",flush=True)
    try:
        server.serve_forever(poll_interval=.2)
    finally:
        stop.set()
        server.server_close()
        supervisor.join(timeout=25)
        if replay_worker:
            replay_worker.terminate()
            try:
                replay_worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                replay_worker.kill()


if __name__ == "__main__":
    main()
