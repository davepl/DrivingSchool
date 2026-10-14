import math
import unittest
import torch
from drivingschool.env import DrivingEnv, OBS_SIZE, TRAIN_TRACKS, HOLDOUT_TRACKS, road, observe
from drivingschool.policy import Policy
from drivingschool.train import gae, evaluate, ppo_loss, Trainer
from drivingschool.live import Bridge
from pathlib import Path
import tempfile


class DrivingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_finite_physics_and_automatic_reset(self):
        env = DrivingEnv(64)
        for _ in range(50):
            obs, reward, done, info, _ = env.step(torch.randn(64,2)*3)
            self.assertEqual(tuple(obs.shape), (64, OBS_SIZE))
            self.assertTrue(torch.isfinite(obs).all())
            self.assertTrue(torch.isfinite(reward).all())
            self.assertTrue(torch.equal(info[:,2].bool(),done))
            self.assertTrue((env.state[:,3]>=0).all())
        env.state[:,0] = 1000
        _, _, done, info, _ = env.step(torch.zeros(64,2))
        self.assertTrue(done.all())
        self.assertTrue((info[:,0]==1).all())
        self.assertTrue((env.state[:,:2].abs()<100).all())

    def test_barrier_collision(self):
        env = DrivingEnv(8, obstacle_on=True)
        env.state[:,:2] = env.obstacles[:,0,:]
        theta = torch.atan2(env.state[:,1],env.state[:,0])
        env.state[:,2] = road(theta,env.params)[2]
        env.state[:,3] = 0
        _, _, done, info, _ = env.step(torch.zeros(8,2))
        self.assertTrue(done.all())
        self.assertTrue((info[:,4]==1).all())

    def test_lap_crossing_angle_wrap(self):
        env = DrivingEnv(8)
        theta = torch.full((8,),-.005)
        radius, _, tangent = road(theta,env.params)
        env.state[:,0] = radius*torch.cos(theta)
        env.state[:,1] = radius*torch.sin(theta)
        env.state[:,2] = tangent
        env.state[:,3] = 10
        env.state[:,5] = math.pi*2-.002
        _, _, done, info, _ = env.step(torch.zeros(8,2))
        self.assertTrue(done.all())
        self.assertTrue((info[:,1]==1).all())
        self.assertTrue((info[:,0]==0).all())

    def test_terminal_gae_does_not_bootstrap(self):
        adv,ret = gae(torch.tensor([[1.],[2.]]),torch.tensor([[False],[True]]),torch.zeros(2,1),torch.tensor([999.]))
        self.assertAlmostEqual(float(adv[1,0]),2)
        self.assertAlmostEqual(float(adv[0,0]),1+.99*.95*2,places=5)

    def test_ppo_updates_policy(self):
        env=DrivingEnv(64)
        policy=Policy(32)
        with torch.no_grad():
            act,log,_=policy.sample(env.observation)
        before=policy.actor.weight.detach().clone()
        loss=ppo_loss(policy,env.observation,act,log,torch.linspace(-1,1,64),torch.ones(64))
        self.assertTrue(torch.isfinite(loss))
        optimizer=torch.optim.Adam(policy.parameters(),lr=.001)
        loss.backward();optimizer.step()
        self.assertFalse(torch.equal(before,policy.actor.weight))

    def test_heldout_tracks_and_repeatable_evaluation(self):
        self.assertTrue(all(t not in TRAIN_TRACKS for t in HOLDOUT_TRACKS))
        policy=Policy(32)
        first=evaluate(policy,"cpu",n=8,seed=7)
        second=evaluate(policy,"cpu",n=8,seed=7)
        self.assertEqual(first,second)

    def test_frozen_validation_checkpoint_keeps_tested_update(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer=Trainer.__new__(Trainer)
            trainer.run_dir=Path(folder)
            trainer.bridge=Bridge()
            trainer.obstacles_enabled=False
            trainer.best_success=-1
            trainer.stage_bests={False:-1,True:-1}
            trainer.update=99
            policy=Policy(32)
            snapshot={"state_dict":policy.state_dict(),"width":32,"update":7,
                "steps":1234,"obstacles":False,"seed":42}
            trainer.accept_evaluation({"lap_success_pct":80},snapshot)
            saved=torch.load(Path(folder)/"best-road.pt",weights_only=True)
            self.assertEqual(saved["update"],7)
            self.assertEqual(saved["evaluation"]["policy_update"],7)
            self.assertTrue(torch.equal(saved["state_dict"]["actor.weight"],policy.actor.weight))

    def test_inference_copy_loads_every_parameter_without_initializing(self):
        original=Policy(32)
        copy=Policy(32,device="meta",initialize=False).to_empty(device="cpu")
        copy.load_state_dict(original.state_dict())
        obs=DrivingEnv(8).observation
        with torch.no_grad():
            self.assertTrue(torch.equal(original(obs)[0],copy(obs)[0]))


if __name__ == "__main__":
    unittest.main()
