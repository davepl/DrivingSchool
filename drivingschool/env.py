"""Tensor-only kinematic bicycle environments. Every car has its own world.

Cars do not collide with one another: overlapping cars in the viewer are ghosted
independent experiences. Training obstacles and road boundaries are real collisions.
Distances are metres; time is simulated seconds. This is a toy driving task, not
a roadworthy autonomous-driving system.
"""
import math
import torch

OBS_SIZE = 18
DT = 0.1
TRAIN_TRACKS = [
    [65, 7, 2, 3, 3, 0.0, 7], [58, 9, 3, 2, 5, 0.3, 7],
    [72, 8, 2, 4, 4, 1.1, 7], [60, 6, 4, 3, 3, 0.8, 7],
    [67, 10, 3, 3, 2, 1.8, 7], [63, 8, 2, 5, 5, 2.3, 7],
]
HOLDOUT_TRACKS = [[69, 9, 4, 4, 3, 0.45, 7], [61, 11, 3, 3, 6, 1.45, 7]]


def wrap(x):
    return torch.remainder(x + math.pi, 2 * math.pi) - math.pi


def road(theta, p):
    r = p[:, 0] + p[:, 1] * torch.cos(p[:, 2] * theta + p[:, 5]) + p[:, 3] * torch.sin(p[:, 4] * theta)
    dr = -p[:, 1] * p[:, 2] * torch.sin(p[:, 2] * theta + p[:, 5]) + p[:, 3] * p[:, 4] * torch.cos(p[:, 4] * theta)
    tangent = theta + math.pi / 2 - torch.atan2(dr, r)
    return r, dr, tangent


def obstacles(p):
    angles = torch.tensor([0.8, 2.8, 4.7], device=p.device, dtype=p.dtype)
    theta = angles.unsqueeze(0).expand(p.shape[0], -1)
    r = p[:, 0:1] + p[:, 1:2] * torch.cos(p[:, 2:3] * theta + p[:, 5:6]) + p[:, 3:4] * torch.sin(p[:, 4:5] * theta)
    r = r + torch.tensor([-2.5, 2.5, -1.8], device=p.device)
    return torch.stack((r * torch.cos(theta), r * torch.sin(theta)), -1)


def observe(state, p, obs_xy, obstacle_on):
    x, y, yaw, speed, steer = state[:, :5].unbind(1)
    theta = torch.atan2(y, x)
    r, dr, tangent = road(theta, p)
    offset = (torch.sqrt(x*x + y*y) - r) / torch.sqrt(1 + (dr/r)**2)
    error = wrap(yaw - tangent)
    features = [offset / p[:, 6], torch.sin(error), torch.cos(error), speed / 25, steer / 0.6, dr / r]
    for ahead in (0.12, 0.3, 0.6):
        _, _, t = road(theta + ahead, p)
        turn = wrap(t - yaw)
        features.extend([torch.sin(turn), torch.cos(turn)])
    delta = obs_xy - state[:, None, :2]
    local_x = delta[:, :, 0] * torch.cos(yaw[:, None]) + delta[:, :, 1] * torch.sin(yaw[:, None])
    local_y = -delta[:, :, 0] * torch.sin(yaw[:, None]) + delta[:, :, 1] * torch.cos(yaw[:, None])
    dist = torch.sqrt(torch.sum(delta * delta, dim=2))
    ranked = torch.where(local_x > -3, dist, torch.full_like(dist, 1000))
    idx = ranked.argmin(1, keepdim=True)
    ox = local_x.gather(1, idx).squeeze(1)
    oy = local_y.gather(1, idx).squeeze(1)
    od = ranked.gather(1, idx).squeeze(1)
    visible = (od < 35) & obstacle_on
    features.extend([torch.where(visible, ox / 35, 1.0), torch.where(visible, oy / 10, 0.0), visible.float()])
    # Include future centre offsets: the policy can anticipate bends without seeing
    # a track ID or absolute position. Held-out courses use the same sensors.
    for ahead in (0.15, 0.4):
        rr, _, _ = road(theta + ahead, p)
        dx = rr * torch.cos(theta + ahead) - x
        dy = rr * torch.sin(theta + ahead) - y
        features.append((-dx * torch.sin(yaw) + dy * torch.cos(yaw)) / 25)
    features.append(torch.ones_like(x))
    return torch.stack(features, dim=1), offset, error, dist.min(1).values


def fresh_state(p, randomness=1.0, generator=None):
    n = len(p)
    theta = torch.rand(n, device=p.device, generator=generator) * (2 * math.pi)
    r, _, tangent = road(theta, p)
    r = r + (torch.rand(n, device=p.device, generator=generator) - 0.5) * 4 * randomness
    yaw = tangent + (torch.rand(n, device=p.device, generator=generator) - 0.5) * 0.8 * randomness
    speed = 5 + torch.rand(n, device=p.device, generator=generator) * 5
    z = torch.zeros_like(theta)
    return torch.stack([r * torch.cos(theta), r * torch.sin(theta), yaw, speed, z, z, z], 1)


def physics_step(state, action, p, obs_xy, obstacle_on):
    x, y, yaw, speed, steer, progress, age = state.unbind(1)
    steer_target = torch.tanh(action[:, 0]) * 0.6
    next_steer = steer + (steer_target - steer) * 0.4
    target_speed = 3 + (torch.tanh(action[:, 1]) + 1) * 11
    next_speed = torch.clamp(speed + torch.clamp(target_speed - speed, -7, 4) * DT, 0, 25)
    next_yaw = wrap(yaw + next_speed / 3.0 * torch.tan(next_steer) * DT)
    next_x = x + next_speed * torch.cos(next_yaw) * DT
    next_y = y + next_speed * torch.sin(next_yaw) * DT
    dtheta = wrap(torch.atan2(next_y, next_x) - torch.atan2(y, x))
    next_progress = progress + dtheta
    next_age = age + 1
    next_state = torch.stack([next_x, next_y, next_yaw, next_speed, next_steer, next_progress, next_age], 1)
    _, offset, error, near_obstacle = observe(next_state, p, obs_xy, obstacle_on)
    offroad = offset.abs() > p[:, 6]
    hit = (near_obstacle < 2.8) & obstacle_on
    wrongway = torch.cos(error) < -0.5
    crash = offroad | hit | wrongway
    lap = next_progress >= 2 * math.pi
    timeout = next_age >= 900
    done = crash | lap | timeout
    # Rewards are based on actual forward travel, staying on road, and avoiding
    # collisions. No steering targets or scripted driver are used for training.
    distance = dtheta * p[:, 0]
    reward = distance * 0.12 - 0.015 * (offset / p[:, 6])**2 - 0.005 * (next_steer - steer)**2
    reward = reward - crash.float() * 3 + lap.float() * 5
    reset = fresh_state(p)
    new_state = torch.where(done[:, None], reset, next_state)
    obs, _, _, _ = observe(new_state, p, obs_xy, obstacle_on)
    # Per-car statistics permit proper aggregation without synchronizing each step.
    info = torch.stack([crash.float(), lap.float(), done.float(), torch.clamp(distance, min=0), hit.float()], 1)
    return new_state, obs, reward, done, info, next_state


class DrivingEnv:
    def __init__(self, n, device="cpu", heldout=False, compiled=False, obstacle_on=False):
        self.n, self.device = n, torch.device(device)
        tracks = HOLDOUT_TRACKS if heldout else TRAIN_TRACKS
        ids = torch.arange(n, device=device) % len(tracks)
        self.params = torch.tensor(tracks, dtype=torch.float32, device=device)[ids]
        self.track_ids = ids
        self.obstacles = obstacles(self.params)
        self.obstacle_on = torch.full((n,), obstacle_on, dtype=torch.bool, device=device)
        self.state = fresh_state(self.params)
        self.observation = observe(self.state, self.params, self.obstacles, self.obstacle_on)[0]
        self.step_fn = torch.compile(physics_step, fullgraph=True) if compiled else physics_step

    def step(self, action):
        self.state, self.observation, reward, done, info, terminal = self.step_fn(
            self.state, action, self.params, self.obstacles, self.obstacle_on)
        return self.observation, reward, done, info, terminal

    def reset(self):
        self.state = fresh_state(self.params)
        self.observation = observe(self.state, self.params, self.obstacles, self.obstacle_on)[0]
        return self.observation

    def set_obstacles(self, enabled):
        self.obstacle_on.fill_(enabled)
        self.observation = observe(self.state, self.params, self.obstacles, self.obstacle_on)[0]
