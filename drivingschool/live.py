"""Human-speed spectator replay and local HTTP UI; training stays on the GPU."""
from contextlib import nullcontext
import json
import threading
import time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
import torch
from .env import DrivingEnv, TRAIN_TRACKS, HOLDOUT_TRACKS, DT
from .policy import Policy


class Bridge:
    def __init__(self):
        self.lock = threading.Lock()
        self.metrics = {"status": "Starting driving school", "steps": 0}
        self.policy_state = None
        self.policy_version = 0
        self.width = 256
        self.checkpoints = {}
        self.paused = False
        self.new_class = threading.Event()
        self.view = {"driver": "latest", "heldout": False, "speed": 1, "stochastic": False}
        self.frame = {}
        self.hardware = {}
        self.history = []

    def record_evaluation(self, update, result):
        with self.lock:
            self.history.append({"update":update, **result})
            self.history = self.history[-300:]

    def update_metrics(self, **kwargs):
        with self.lock:
            self.metrics.update(kwargs)

    def set_policy(self, policy, width):
        state = {k:v.detach().cpu().clone() for k,v in policy.state_dict().items()}
        with self.lock:
            self.policy_state, self.width = state, width
            self.policy_version += 1

    def register_checkpoint(self, path, summary):
        key = path.parent.name + "/" + path.stem
        with self.lock:
            self.checkpoints[key] = {"path": str(path), "key": key, **summary}

    def snapshot(self):
        with self.lock:
            return {"metrics": dict(self.metrics), "frame": self.frame, "hardware": self.hardware,
                    "view": dict(self.view), "paused": self.paused,
                    "history":list(self.history),
                    "checkpoints": [{k:v for k,v in c.items() if k != "path"} for c in self.checkpoints.values()]}


def spectator(bridge, stop, count=384, device="cpu"):
    device=torch.device(device)
    stream=torch.cuda.Stream(device=device,priority=-1) if device.type=="cuda" else None
    def work_stream():
        return torch.cuda.stream(stream) if stream else nullcontext()
    policy, previous_view, previous_run = None, None, None
    event_seq = 0
    last_weights = None
    total_crashes = total_laps = 0
    while not stop.is_set():
        begin = time.monotonic()
        with bridge.lock:
            view = dict(bridge.view)
            latest = bridge.policy_state
            current_version = bridge.policy_version
            width = bridge.width
            checkpoints = dict(bridge.checkpoints)
            obstacles = bridge.metrics.get("obstacles", False)
            run = bridge.metrics.get("run", "")
        if latest is None:
            stop.wait(.05)
            continue
        if view != previous_view or run != previous_run:
            env = DrivingEnv(count, "cpu", heldout=view["heldout"], obstacle_on=obstacles)
            previous_view = view
            previous_run = run
            last_weights = None
            total_crashes = total_laps = 0
        if view["driver"] == "latest":
            weights_key = ("latest", current_version)
            weights = latest
            metadata = {"width": width, "obstacles": obstacles, "update": bridge.metrics.get("update", 0),
                        "evaluation": bridge.metrics.get("evaluation")}
        else:
            key = view["driver"]
            if key == "rookie":
                key = run + "/rookie"
            checkpoint = checkpoints.get(key)
            weights_key = (key, checkpoint.get("update") if checkpoint else None)
            if checkpoint and weights_key != last_weights:
                saved = torch.load(checkpoint["path"], map_location="cpu", weights_only=True)
                weights, metadata = saved["state_dict"], saved
            elif checkpoint:
                weights = None
            else:
                weights_key = ("latest", current_version)
                weights = latest
                metadata = {"width": width, "obstacles": obstacles, "update": 0}
        if weights_key != last_weights:
            with work_stream():
                if policy is None or policy.width != metadata["width"]:
                    # Checkpoints replace every tensor; avoid expensive orthogonal
                    # initialization for the large-policy spectator copy.
                    policy=Policy(metadata["width"],device="meta",initialize=False).to_empty(device=device)
                policy.load_state_dict(weights)
            policy.eval()
            env.set_obstacles(metadata.get("obstacles", False))
            last_weights = weights_key
        events = []
        with torch.inference_mode():
            for _ in range(view["speed"]):
                with work_stream(), torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
                    observation=env.observation.to(device)
                    if view["stochastic"] or view["driver"] == "rookie":
                        actions=policy.sample(observation)[0]
                    else:
                        actions=policy(observation)[0]
                    actions=actions.float().cpu()
                _, _, done, info, terminal = env.step(actions)
                crashed = torch.nonzero(info[:, 0]).flatten()
                for i in crashed[:64].tolist():
                    event_seq += 1
                    events.append([event_seq, int(env.track_ids[i]), round(float(terminal[i,0]),2), round(float(terminal[i,1]),2)])
                total_crashes += int(info[:, 0].sum())
                total_laps += int(info[:, 1].sum())
        frame = {"seq": event_seq, "time": time.time(), "run":run,"cars": env.state[:,:5].tolist(),
                 "track_ids": env.track_ids.tolist(), "tracks": HOLDOUT_TRACKS if view["heldout"] else TRAIN_TRACKS,
                 "events": events, "crashes": total_crashes, "laps": total_laps,
                 "count": count, "replay_device":str(device), "frame_seconds":time.monotonic()-begin,
                 "policy_update":metadata.get("update",0), "policy_evaluation":metadata.get("evaluation"),
                 "obstacles": bool(env.obstacle_on[0]), "obstacle_xy": env.obstacles[:len(HOLDOUT_TRACKS if view['heldout'] else TRAIN_TRACKS)].tolist()}
        with bridge.lock:
            bridge.frame = frame
        stop.wait(max(0, DT - (time.monotonic()-begin)))


def hardware_monitor(bridge, stop):
    import subprocess
    while not stop.is_set():
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,power.draw,temperature.gpu,power.limit", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3)
            rows = []
            for line in result.stdout.strip().splitlines():
                name, util, mem, watts, temp, limit = [v.strip() for v in line.split(",")]
                rows.append({"name": name, "utilization": float(util), "memory_gb": float(mem)/1024, "watts": float(watts), "temperature": float(temp), "power_limit_watts":float(limit)})
            with bridge.lock:
                bridge.hardware = {"gpus": rows, "updated": time.time()}
        except Exception as error:
            with bridge.lock:
                bridge.hardware = {"error": str(error)}
        stop.wait(3)


def serve(bridge, web_root, host, port):
    web_root = Path(web_root)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send_json(self, payload, code=200):
            data = json.dumps(payload, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/api/state":
                return self.send_json(bridge.snapshot())
            if path == "/api/health":
                return self.send_json({"ok": True, "status": bridge.metrics.get("status")})
            if path == "/api/replay-config":
                if self.client_address[0] not in ("127.0.0.1","::1"):
                    return self.send_error(403)
                with bridge.lock:
                    view=dict(bridge.view)
                    run=bridge.metrics.get("run","")
                    key=run+"/latest" if view["driver"]=="latest" else run+"/rookie" if view["driver"]=="rookie" else view["driver"]
                    checkpoint=bridge.checkpoints.get(key) or bridge.checkpoints.get(run+"/rookie")
                    config={"view":view,"metrics":dict(bridge.metrics),"checkpoint":checkpoint}
                return self.send_json(config)
            files = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/style.css": "style.css"}
            filename = files.get(path)
            if not filename:
                return self.send_error(404)
            data = (web_root / filename).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", {"html":"text/html; charset=utf-8", "js":"text/javascript", "css":"text/css"}[filename.rsplit('.',1)[1]])
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            # Default listener is loopback. Reject cross-origin browser requests
            # and non-loopback control requests even if viewing is bound to LAN.
            if self.client_address[0] not in ("127.0.0.1", "::1"):
                return self.send_json({"error":"Controls require a browser on the Dell or an SSH tunnel"},403)
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc != self.headers.get("Host"):
                return self.send_json({"error":"Cross-origin controls are disabled"},403)
            path=urlparse(self.path).path
            if path not in ("/api/control","/api/replay-frame"):
                return self.send_error(404)
            try:
                length = int(self.headers.get("Content-Length", 0))
                if length > (256*1024 if path=="/api/replay-frame" else 4096):
                    raise ValueError("Request too large")
                request = json.loads(self.rfile.read(length))
                if path=="/api/replay-frame":
                    frame=request["frame"]
                    if len(frame.get("cars",[]))>4096:
                        raise ValueError("Replay sample too large")
                    with bridge.lock:
                        if request["run"]==bridge.metrics.get("run"):
                            bridge.frame=frame
                    return self.send_json({"ok":True})
                command = request["command"]
                with bridge.lock:
                    if command == "pause":
                        bridge.paused = True
                    elif command == "resume":
                        bridge.paused = False
                    elif command == "new_class":
                        bridge.new_class.set()
                    elif command == "view":
                        driver = request.get("driver", bridge.view["driver"])
                        if driver not in ("latest", "rookie") and driver not in bridge.checkpoints:
                            raise ValueError("Unknown checkpoint")
                        speed = int(request.get("speed", bridge.view["speed"]))
                        if speed not in (1,2,4):
                            raise ValueError("Speed must be 1, 2, or 4")
                        bridge.view = {"driver":driver, "speed":speed,
                            "heldout": bool(request.get("heldout", bridge.view["heldout"])),
                            "stochastic":bool(request.get("stochastic", bridge.view["stochastic"]))}
                    else:
                        raise ValueError("Unknown command")
                self.send_json({"ok": True})
            except (ValueError, KeyError, TypeError) as error:
                self.send_json({"error":str(error)},400)
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server
