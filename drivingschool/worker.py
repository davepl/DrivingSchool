"""Isolated RTX process for replay/validation while the GB300 trains continuously."""
import argparse
import json
import threading
import time
import urllib.request
from pathlib import Path
import torch
from .policy import Policy
from .train import evaluate
from .live import Bridge,spectator


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("role",choices=("replay","evaluate"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--port",type=int,default=8780)
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    torch.cuda.set_device(0)
    if args.role=="evaluate":
        snapshot=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
        policy=Policy(snapshot["width"],device="meta",initialize=False).to_empty(device="cuda:0")
        policy.load_state_dict(snapshot["state_dict"])
        result=evaluate(policy,"cuda:0",snapshot["obstacles"])
        print(json.dumps(result),flush=True)
        return
    bridge=Bridge()
    stop=threading.Event()
    threading.Thread(target=spectator,args=(bridge,stop,384,"cuda:0"),daemon=True).start()
    base=f"http://127.0.0.1:{args.port}"
    loaded=None
    posted_time=0
    while True:
        start=time.monotonic()
        try:
            with urllib.request.urlopen(base+"/api/replay-config",timeout=2) as response:
                config=json.load(response)
            checkpoint=config.get("checkpoint")
            if not checkpoint:
                time.sleep(.1)
                continue
            key=(checkpoint["path"],checkpoint["update"])
            if key!=loaded:
                saved=torch.load(checkpoint["path"],map_location="cpu",weights_only=True)
                with bridge.lock:
                    bridge.policy_state=saved["state_dict"]
                    bridge.width=saved["width"]
                    bridge.policy_version+=1
                bridge.register_checkpoint(Path(checkpoint["path"]),{k:v for k,v in saved.items() if k!="state_dict"})
                loaded=key
            with bridge.lock:
                bridge.view=config["view"]
                bridge.metrics=dict(config["metrics"],update=checkpoint["update"])
                frame=bridge.frame
            if frame and frame["time"]!=posted_time:
                frame=dict(frame,replay_device=torch.cuda.get_device_name(0))
                data=json.dumps({"run":frame.get("run",config["metrics"]["run"]),"frame":frame}).encode()
                request=urllib.request.Request(base+"/api/replay-frame",data=data,headers={"Content-Type":"application/json"})
                with urllib.request.urlopen(request,timeout=2) as response:
                    response.read()
                posted_time=frame["time"]
        except (OSError,ValueError,KeyError) as error:
            # Startup/restart can briefly remove the API or replace a checkpoint.
            print(f"Replay waiting: {error}",flush=True)
        time.sleep(max(.01,.1-(time.monotonic()-start)))


if __name__=="__main__":
    main()
