"""
    Training a PPO agent.
"""

from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO

from marlos.custom_envs.random_off_switch import RandomOffSwitchEnv
from marlos.training.classes import MinigridFeaturesExtractor


def main():
    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )

    env = RandomOffSwitchEnv(render_mode="rgb_array")
    env = ImgObsWrapper(env)

    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(2e5)


if __name__ == "__main__":
    main()
