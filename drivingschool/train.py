"""PPO training, measured throughput, deterministic held-out evaluation."""
import json
import time
from pathlib import Path
import torch
from .env import DrivingEnv, OBS_SIZE, DT, fresh_state, observe
from .policy import Policy


@torch.no_grad()
def evaluate(policy, device, obstacles=False, n=128, seed=12345):
    # A private generator fixes test starts without changing training RNG state.
    with torch.no_grad():
        env = DrivingEnv(n, device, heldout=True, obstacle_on=obstacles)
        rng = torch.Generator(device=device).manual_seed(seed)
        env.state = fresh_state(env.params, generator=rng)
        env.observation = observe(env.state, env.params, env.obstacles, env.obstacle_on)[0]
        alive = torch.ones(n, dtype=torch.bool, device=device)
        successes = torch.zeros(n, device=device)
        survived = torch.zeros(n, device=device)
        distance = torch.zeros(n, device=device)
        for _ in range(900):
            action = policy(env.observation)[0]
            _, _, done, info, _ = env.step(action)
            successes += info[:, 1] * alive
            survived += alive.float()
            distance += info[:, 3] * alive
            alive &= ~done
            if not alive.any().item():
                break
        return {"lap_success_pct": round(successes.mean().item()*100, 1),
                "mean_survival_seconds": round(survived.mean().item()*DT, 1),
                "mean_distance_m": round(distance.mean().item(), 1), "cars": n,
                "seed": seed, "obstacles": obstacles}


def gae(rewards, dones, values, last_value):
    advantages = torch.empty_like(rewards)
    carry = torch.zeros_like(last_value)
    next_value = last_value
    for step in reversed(range(len(rewards))):
        not_done = (~dones[step]).float()
        delta = rewards[step] + .99 * next_value * not_done - values[step]
        carry = delta + .99 * .95 * not_done * carry
        advantages[step] = carry
        next_value = values[step]
    return advantages, advantages + values


def ppo_loss(policy, obs, action, old_logprob, advantages, returns):
    logprob, entropy, value = policy.score(obs, action)
    ratio = (logprob - old_logprob).exp()
    clipped = ratio.clamp(.8, 1.2)
    actor = -torch.minimum(ratio * advantages, clipped * advantages).mean()
    critic = .5 * (value - returns).square().mean()
    return actor + .5 * critic - .004 * entropy


class Trainer:
    def __init__(self, args, bridge, run_dir, stop):
        self.args, self.bridge, self.stop = args, bridge, stop
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = args.device
        torch.manual_seed(args.seed)
        self.policy = Policy(args.width).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4, eps=1e-5, fused=str(self.device).startswith("cuda"))
        self.env = DrivingEnv(args.envs, self.device, compiled=not args.eager)
        self.sample = torch.compile(self.policy.sample, fullgraph=True) if not args.eager else self.policy.sample
        self.loss_fn = torch.compile(ppo_loss, fullgraph=True) if not args.eager else ppo_loss
        self.advantages_fn = torch.compile(gae, fullgraph=True) if not args.eager else gae
        self.update = self.steps = self.crashes = self.laps = 0
        self.metres = self.sim_seconds = 0.0
        self.obstacles_enabled = False
        self.best_success = -1
        self.stage_bests = {False: -1, True: -1}
        self.started = time.monotonic()
        self.baseline = evaluate(self.policy, self.device)
        self.last_eval = self.baseline
        self.bridge.record_evaluation(0, self.baseline)
        self.save("rookie", self.baseline)
        self.bridge.set_policy(self.policy, args.width)
        self.bridge.update_metrics(status="Warming up GPU kernels", baseline=self.baseline,
            evaluation=self.last_eval, run=self.run_dir.name, envs=args.envs,
            device=torch.cuda.get_device_name(self.device) if str(self.device).startswith("cuda") else "CPU",
            seed=args.seed, steps=0, update=0, obstacles=False, width=args.width)

    def save(self, label, evaluation=None):
        filename = self.run_dir / (label + ".pt")
        payload = {"state_dict": {k:v.detach().cpu() for k,v in self.policy.state_dict().items()},
                   "width": self.args.width, "update": self.update, "steps": self.steps,
                   "obstacles": self.obstacles_enabled, "evaluation": evaluation,
                   "seed": self.args.seed}
        tmp = filename.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(filename)
        summary = {k:v for k,v in payload.items() if k != "state_dict"}
        filename.with_suffix(".json").write_text(json.dumps(summary, indent=2))
        self.bridge.register_checkpoint(filename, summary)

    def run(self):
        n, horizon = self.args.envs, self.args.horizon
        shape = (horizon, n)
        obs_buf = torch.empty((*shape, OBS_SIZE), device=self.device)
        act_buf = torch.empty((*shape, 2), device=self.device)
        log_buf = torch.empty(shape, device=self.device)
        val_buf = torch.empty(shape, device=self.device)
        rew_buf = torch.empty(shape, device=self.device)
        done_buf = torch.empty(shape, dtype=torch.bool, device=self.device)
        metrics_file = self.run_dir / "metrics.jsonl"
        while not self.stop.is_set() and (not self.args.updates or self.update < self.args.updates):
            if self.bridge.paused:
                self.bridge.update_metrics(status="Training paused — live driving continues")
                self.stop.wait(.2)
                continue
            begin = time.monotonic()
            stats = torch.zeros(5, device=self.device)
            with torch.no_grad():
                for t in range(horizon):
                    obs_buf[t].copy_(self.env.observation)
                    action, logprob, value = self.sample(self.env.observation)
                    act_buf[t].copy_(action)
                    log_buf[t].copy_(logprob)
                    val_buf[t].copy_(value)
                    _, reward, done, info, _ = self.env.step(action)
                    rew_buf[t].copy_(reward)
                    done_buf[t].copy_(done)
                    stats += info.sum(0)
                last_value = self.policy(self.env.observation)[1]
                advantages, returns = self.advantages_fn(rew_buf, done_buf, val_buf, last_value)
                advantages = (advantages - advantages.mean()) / (advantages.std()+1e-8)
            flat_obs, flat_act = obs_buf.reshape(-1, OBS_SIZE), act_buf.reshape(-1, 2)
            flat_log, flat_adv, flat_ret = log_buf.flatten(), advantages.flatten(), returns.flatten()
            losses = []
            for epoch in range(self.args.epochs):
                order = torch.randperm(horizon*n, device=self.device)
                for start in range(0, horizon*n, self.args.batch):
                    idx = order[start:start+self.args.batch]
                    self.optimizer.zero_grad(set_to_none=True)
                    loss = self.loss_fn(self.policy, flat_obs[idx], flat_act[idx], flat_log[idx], flat_adv[idx], flat_ret[idx])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), .5)
                    self.optimizer.step()
                    losses.append(loss.detach())
            if str(self.device).startswith("cuda"):
                torch.cuda.synchronize(self.device)
            duration = time.monotonic() - begin
            self.update += 1
            self.steps += horizon*n
            numbers = stats.cpu().tolist()
            self.crashes += int(numbers[0])
            self.laps += int(numbers[1])
            self.metres += numbers[3]
            self.sim_seconds += horizon*n*DT
            self.bridge.set_policy(self.policy, self.args.width)
            evaluation_seconds = 0
            if self.update % self.args.eval_every == 0 or self.update == 1:
                self.bridge.update_metrics(status="Testing on two unseen tracks")
                ev_start = time.monotonic()
                self.last_eval = evaluate(self.policy, self.device, self.obstacles_enabled)
                self.bridge.record_evaluation(self.update, self.last_eval)
                evaluation_seconds = time.monotonic() - ev_start
                self.save(f"lesson-{self.update:04d}", self.last_eval)
                if self.last_eval["lap_success_pct"] > self.best_success:
                    self.best_success = self.last_eval["lap_success_pct"]
                    self.save("best", self.last_eval)
                if self.last_eval["lap_success_pct"] > self.stage_bests[self.obstacles_enabled]:
                    self.stage_bests[self.obstacles_enabled] = self.last_eval["lap_success_pct"]
                    self.save("best-barriers" if self.obstacles_enabled else "best-road", self.last_eval)
            self.save("latest", self.last_eval)
            metrics = dict(status="Learning with PPO", run=self.run_dir.name, envs=n, steps=self.steps,
                update=self.update, crashes=self.crashes, laps=self.laps, miles=self.metres/1609.344,
                simulated_hours=self.sim_seconds/3600, steps_per_second=horizon*n/duration,
                update_seconds=duration, evaluation_seconds=evaluation_seconds,
                loss=torch.stack(losses).mean().item(), wall_seconds=time.monotonic()-self.started,
                overall_steps_per_second=self.steps/(time.monotonic()-self.started),
                evaluation=self.last_eval, baseline=self.baseline, obstacles=self.obstacles_enabled,
                gpu_memory_gb=torch.cuda.max_memory_allocated(self.device)/1e9 if str(self.device).startswith("cuda") else 0,
                actual_population_sample=self.env.state[:32,:5].cpu().tolist())
            self.bridge.update_metrics(**metrics)
            with metrics_file.open("a") as f:
                f.write(json.dumps(metrics) + "\n")
            print(f"Update {self.update:4d} | {metrics['steps_per_second']:,.0f} car-steps/s | "
                  f"{self.last_eval['lap_success_pct']:.1f}% held-out laps | {self.crashes:,} crashes", flush=True)
            # Introduce fixed barriers only after the driver can reliably complete
            # unfamiliar tracks. Evaluation clearly records which lesson is tested.
            if not self.obstacles_enabled and self.last_eval["lap_success_pct"] >= 80 and self.update >= 20:
                self.obstacles_enabled = True
                self.env.set_obstacles(True)
                self.best_success = -1
                self.bridge.update_metrics(obstacles=True, status="New lesson: avoid barriers")
        self.save("latest", self.last_eval)
        self.bridge.update_metrics(status="Class complete — best driver is on track", complete=True)
        best_key = self.run_dir.name + ("/best-barriers" if self.obstacles_enabled else "/best-road")
        with self.bridge.lock:
            if self.bridge.view["driver"] == "latest" and best_key in self.bridge.checkpoints:
                self.bridge.view["driver"] = best_key
