import torch
import torch.nn as nn
from pettingzoo.classic import blackjack_v1
from tensordict import TensorDict
from torch.distributions import Categorical
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import ParallelEnv, PettingZooEnv
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.modules.tensordict_module import SafeModule
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

from environment import BlackjackEnv

# Use PettingZooEnv wrapper instead of ParallelEnv

# Hyperparameters
lr = 3e-4
num_epochs = 1000
frames_per_batch = 1000
num_envs = 4
gae_lambda = 0.95
gamma = 0.99
clip_epsilon = 0.2
entropy_eps = 1e-4

# Create PettingZoo environment
pz_env = blackjack_v1.env()
env = PettingZooEnv(pz_env, device="cpu")
env = env.expand(num_envs)


# Network architecture
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.network(x)


class ValueNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.network(x)


# Get observation and action specs
obs_spec = env.observation_spec
action_spec = env.action_spec

# Create policy and value networks
policy_net = PolicyNetwork(obs_spec.shape[-1], 128, action_spec.space.n)
value_net = ValueNetwork(obs_spec.shape[-1], 128)

# Create TorchRL modules
policy_module = SafeModule(policy_net, in_keys=["observation"], out_keys=["logits"])

value_module = SafeModule(value_net, in_keys=["observation"], out_keys=["state_value"])

# Create probabilistic actor
actor = ProbabilisticActor(
    module=policy_module,
    distribution_class=Categorical,
    in_keys=["logits"],
    out_keys=["action"],
    distribution_kwargs={"logits": "logits"},
)

# Create value operator
critic = ValueOperator(module=value_module, in_keys=["observation"])

# Create data collector
collector = SyncDataCollector(
    env, actor, frames_per_batch=frames_per_batch, total_frames=-1, device="cpu"
)

# Create replay buffer
replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(frames_per_batch), sampler=SamplerWithoutReplacement()
)

# Create GAE estimator
advantage_module = GAE(
    gamma=gamma, lmbda=gae_lambda, value_network=critic, average_gae=True
)

# Create PPO loss
loss_module = ClipPPOLoss(
    actor=actor,
    critic=critic,
    clip_epsilon=clip_epsilon,
    entropy_bonus=True,
    entropy_coef=entropy_eps,
    normalize_advantage=True,
)

# Optimizers
actor_optim = torch.optim.Adam(actor.parameters(), lr=lr)
critic_optim = torch.optim.Adam(critic.parameters(), lr=lr)

# Training loop
pbar = tqdm(range(num_epochs))
for epoch in pbar:
    # Collect data
    for i, data in enumerate(collector):
        if i >= 1:  # Collect one batch
            break

    # Compute advantages
    with torch.no_grad():
        data = advantage_module(data)

    # Store in replay buffer
    replay_buffer.extend(data.view(-1))

    # Training step
    for _ in range(10):  # Multiple updates per data collection
        sample = replay_buffer.sample()
        loss_vals = loss_module(sample)

        actor_loss = loss_vals["loss_objective"] + loss_vals["loss_entropy"]
        critic_loss = loss_vals["loss_critic"]

        actor_optim.zero_grad()
        actor_loss.backward()
        actor_optim.step()

        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()

    # Update progress bar
    if epoch % 100 == 0:
        pbar.set_description(
            f"Epoch {epoch}, Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}"
        )

print("Training completed!")

# Save models
torch.save(actor.state_dict(), "blackjack_actor.pth")
torch.save(critic.state_dict(), "blackjack_critic.pth")
