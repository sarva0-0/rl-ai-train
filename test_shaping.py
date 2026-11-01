"""Check that the shaping term is potential-based."""
import gymnasium as gym
import pytest

from train import ShapedLander


def test_shaping_is_potential_based():
    env = ShapedLander(gym.make("LunarLander-v3"))
    g = env.gamma
    obs, _ = env.reset(seed=0)
    phi0 = env._potential(obs)
    total, t, done = 0.0, 0, False
    while not done:
        obs, _, term, trunc, info = env.step(0)
        total += g**t * info["shaping"]
        t += 1
        done = term or trunc
    assert total == pytest.approx(g**t * env._potential(obs) - phi0, abs=0.05)
