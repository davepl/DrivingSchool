"""Sample real hardware load and viewer latency while the demo trains."""
import argparse
import json
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument("--seconds",type=int,default=60)
parser.add_argument("--port",type=int,default=8780)
parser.add_argument("--output",default="benchmarks/dell-2026-10-14/max-load")
args=parser.parse_args()
prefix=Path(args.output)
prefix.parent.mkdir(parents=True,exist_ok=True)
rows=[]
for _ in range(args.seconds):
    start=time.monotonic()
    result=subprocess.run(["nvidia-smi","-i","1","--query-gpu=utilization.gpu,power.draw,power.limit,memory.used,temperature.gpu","--format=csv,noheader,nounits"],capture_output=True,text=True,check=True)
    util,watts,limit,memory,temp=map(float,result.stdout.strip().split(","))
    with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/api/state",timeout=3) as response:
        state=json.load(response)
    frame=state["frame"]
    rows.append({"time":time.time(),"utilization_pct":util,"watts":watts,"limit_watts":limit,
        "memory_mib":memory,"temperature_c":temp,"update":state["metrics"].get("update"),
        "viewer_frame_age_seconds":time.time()-frame.get("time",0),
        "viewer_compute_seconds":frame.get("frame_seconds"),
        "population":state["metrics"].get("envs"),"width":state["metrics"].get("width")})
    time.sleep(max(0,1-(time.monotonic()-start)))
summary={"samples":len(rows),"minimum_utilization_pct":min(r["utilization_pct"] for r in rows),
    "mean_utilization_pct":statistics.mean(r["utilization_pct"] for r in rows),
    "samples_at_100_pct":sum(r["utilization_pct"]==100 for r in rows),
    "mean_watts":statistics.mean(r["watts"] for r in rows),"peak_watts":max(r["watts"] for r in rows),
    "peak_temperature_c":max(r["temperature_c"] for r in rows),
    "mean_viewer_frame_age_seconds":statistics.mean(r["viewer_frame_age_seconds"] for r in rows),
    "population":rows[0]["population"],"width":rows[0]["width"],
    "first_update":rows[0]["update"],"last_update":rows[-1]["update"],
    "notes":"Whole-GB300 readings; includes the existing Robotron process. Viewer/validation run on isolated RTX processes."}
prefix.with_suffix(".jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
prefix.with_suffix(".json").write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
