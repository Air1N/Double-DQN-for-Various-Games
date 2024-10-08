from collections import deque, namedtuple
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import gymnasium as gym
from tqdm import tqdm
from utils.multiplot import Multiplot
from utils.memory_stack import MemoryStack
from utils.dqn_utils import GreedyEpsilon, ModelAdjuster

# Set the environment name. This model is currently tested on CartPole-v1
environment_name = 'LunarLander-v2'
env = gym.make(environment_name, render_mode='human')

# Choose device automatically
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# Seed for consistency in comparisons
torch.manual_seed(123)

print(f"device: {device}")

# Model step to load, this is the number at the end of the file name @ 0 no file is loaded. [Default: 0]
load_step = 0

# This is here because it's appended to the name of the save file, it counts up by 1 each frame. [Default: load_step=0]
step = load_step

# Path to the model file to load. Automatically generated based on step and environment_name values.
if load_step > 0:
    model_to_load = f"models/{environment_name}/actor_model_{step}.pth"
else: model_to_load = ""

BATCH_SIZE = 64 # The number of transitions per mini-batch [Default: 64]
INPUT_N_STATES = 4 # The number of consecutive states to be concatenated for the observation/input. [Default: 4]

TRAIN_INTERVAL = 1 # The number of frames between each training step. [Default: 1]
SAVE_INTERVAL = 5000 # The number of frames between saving the model to a file. [Default: 500]

EPOCHS = 5000 # This determines the maximum length the program will run for, in epochs. [Default: 500]
EPISODES_PER_EPOCH = 1 # This determines how many episodes, playing until termination, there are in each epoch. [Default: 10]

SNAPSHOT_INTERVAL = 1 # The number of epochs between showing the human visualization. [Default: 25]
SHOW_FIRST = True # Regardless of snapshot interval, epoch 0 won't show a visualization, unless this is TRUE. [Default: False]

SOFT_COPY_INTERVAL = 1 # Number of steps before doing a soft-copy. pred_model.params += actor_model.params * TAU. [Default: 1]
HARD_COPY_INTERVAL = 10000 # Number of steps before doing a hard-copy. pred_model = actor_model. [Default: 10000]

GAMMA = 0.99 # Affects how much the model takes into account future Q-values in the current state. target_output = reward + GAMMA * pred_model(next_state)[actor_model(next_state).argmax()] -- Standard DDQN implementation
TAU = 0.0001 # Affects the speed of parameter transfer during soft-copy. pred_model.params += actor_model.params * TAU. High numbers result in instability. [Default: 0.0001]

ACTOR_LR = 0.00015 # Learning rate used in the optimizer. [Default: 0.00015]

REWARD_SCALING = 25 # These are for use more complex reward-shape problems. [Default: +25]
MIN_REWARD = -1 # These are for use in more complex reward-shape problems. [Default: -1]
MAX_REWARD = 1 # These are for use in more complex reward-shape problems. [Default: +1]

REWARD_AFFECT_PAST_N = 10 # Affect how many previous reward states, each with diminishing effects. [Default: 4]
REWARD_AFFECT_THRESH = [-5, 5] # At what thresholds does the reward propogate to the previous samples? [Default: [-0.8, 2]]

MEMORY_REWARD_THRESH = 0.00 # Assume  anything with less abs(reward) isn't useful to learn, and exclude it from memory [Default: 0.04]

DISABLE_RANDOM = False # Disable epsilon_greedy exploration function. [Default: False]
SAVING_ENABLED = True # Enable saving of model files. [Default: True]
LEARNING_ENABLED = True # Enable model training. [Default: True]

eps = 2 # Starting epsilon value, used in the epsilon_greedy policy. [Default: 0.5]
EPS_DECAY = 0.001 # How much epsilon decays each time a random action is chosen. [Default: 0.0001]
MIN_EPS = 0.05 # Minimum epsilon/random action chance. Keep this above 0 to encourage continued learning. [Default: 0.01]

# Surprisal is calculated by taking the sum(abs(next_state_batch - next_state_guess)**exponent)
SURPRISAL_EXPONENT = 1 # TODO The exponent applied to individual differences in next state guess. Essentially, how influential are outliers. [Default: 2]
SURPRISAL_BIAS = 0 # Bias the surprisal score before weighting [Default: -1]
SURPRISAL_WEIGHT = 0.001 # The amount that surprisal influences the reward function. [Default: 0.01]

plt.ion()

multiplot = Multiplot(names=("a_loss", "rb", "real_reward", "cumulative_reward", "natural_reward", "cb", "surprisal", "grad_norm", "rb", "output_0", "output_1", "output_2", "output_3"))
greedy_epsilon = GreedyEpsilon(DISABLE_RANDOM, EPS_DECAY, MIN_EPS)
model_adjuster = ModelAdjuster(TAU, HARD_COPY_INTERVAL, SOFT_COPY_INTERVAL)

# torch.autograd.set_detect_anomaly(True)
torch.set_printoptions(2, sci_mode=False)

class CustomDQN(torch.nn.Module):
    """
    This class creates a pytorch DQN with a predetermined structure.

    Attributes:
        isPred (boolean): Whether the model is the prediction model or not.

        self.lin_1 (nn.Linear): Shared input layer.

        self.lin_2a (nn.Linear): Hidden layer for Q-value prediction.
        self.lin_oA (nn.Linear): Output layer for Q-value prediction.

        self.lin_2b (nn.Linear): Hidden layer for next state prediction.
        self.lin_oB (nn.Linear): Output layer for next state prediction.
    """
    def __init__(self, isPred):
        """
        The constructor for the CustomDQN class.

        Parameters:
            isPred (boolean): Whether the model is the prediction model or not.
        """
        super(CustomDQN, self).__init__()

        self.isPred = isPred

        self.lin_1 = nn.Linear(env.observation_space.shape[0] * INPUT_N_STATES, 64)

        self.lin_2a = nn.Linear(64, 64)
        self.lin_oA = nn.Linear(64, env.action_space.n)

        self.lin_2b = nn.Linear(64 + env.action_space.n, 64)
        self.lin_oB = nn.Linear(64, env.observation_space.shape[0] * INPUT_N_STATES)

    def forward(self, x, real_actions=None, training=False):
        global eps
        """
        The feed-forward/step function of the model.

        Parameters:
            x (torch.tensor): The input state tensor for the model.
            real_actions (torch.tensor): A batch of real actions the model took, only used in training.
            training (boolean): Enable training-specific changes. i.e. Disables greedy-epsilon.
        
        Returns:
            tuple (a, b):
                - a (torch.tensor): The output action Q-values.
                - b (torch.tensor): The predicted next state.
        """

        x = F.leaky_relu(self.lin_1(x)) # Take state as input and run through 1 linear layer

        # First head predicts Q values for actions
        a = F.leaky_relu(self.lin_2a(x)) 
        a = self.lin_oA(a)

        explore, eps = greedy_epsilon.choose(eps)
        if not training and explore:
            a = torch.rand_like(a) * 2 - 1
        
        chosen_actions = torch.argmax(a, dim=1)

        # During training, the action is not taken.
        # Fortunately, an action was already taken in that state and saved. Those saved actions can be used here.
        if real_actions != None:
            chosen_actions = real_actions

        one_hot_encoded_action = torch.zeros_like(a).scatter_(1, chosen_actions.unsqueeze(1), 1.)
        
        # Second head predicts next state from state + Q-values
        b = torch.cat((x, one_hot_encoded_action), dim=1)
        b = F.leaky_relu(self.lin_2b(b))
        b = self.lin_oB(b)

        return a, b

actor_model = CustomDQN(isPred=False)
if model_to_load != "":
    actor_model = torch.load(model_to_load)

pred_model = CustomDQN(isPred=True)
pred_model.load_state_dict(actor_model.state_dict())
pred_model.eval()

actor_optimizer = torch.optim.RAdam(actor_model.parameters(), lr=ACTOR_LR)

actor_model.to(device)
pred_model.to(device)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

actor_mem = MemoryStack(1000000)

def try_learning():
    """
    Perform checks and start `model_train()`.
    """
    global step

    if not LEARNING_ENABLED: return

    if len(actor_mem.memory) > BATCH_SIZE:
        if step % TRAIN_INTERVAL == 0:
            a_loss = model_train(BATCH_SIZE)
            multiplot.add_entry('a_loss', a_loss.cpu().detach().numpy())


    if step % SAVE_INTERVAL == 0 and SAVING_ENABLED:
        torch.save(actor_model, f"models/{environment_name}/actor_model_{step}.pth")


short_memory = []

def affect_short_mem(reward):
    """
    Alter the n=`REWARD_AFFECT_PAST_N` most recent `short_memory` reward values before they're passed into the MemoryStack.

    Parameters:
        reward (float): This value is compared against `MEMORY_REWARD_THRESH` and if its absolute value is higher, then apply the reward to the previous `REWARD_AFFECT_PAST_N` states. The effect is diminished for less recent samples.
    """
    global short_memory

    # If short_memory is long enough:
    if len(short_memory) > REWARD_AFFECT_PAST_N:
        send_short_to_long_mem(1)

    # Only apply if the current reward exceeds a threshold. 
    # Affect short_memory reward values based on reward recieved currently, diminishing for less recent events.
    if reward < REWARD_AFFECT_THRESH[0] or reward > REWARD_AFFECT_THRESH[1]:
        for i in range(0, len(short_memory)):
            short_memory[-(i + 1)][3] += reward / (i + 1)

def send_short_to_long_mem(n):
    """
    Sends the oldest `n` elements from short_memory to actor_mem.

    Parameters:
        n (int): The number of elements to send from short_memory to actor_mem.
    """
    for i in range(0, n):
        # Remove the first element
        short_mem = Transition(*short_memory.pop(0))

        # Log it as real reward
        multiplot.add_entry('real_reward', short_mem.reward.cpu().detach().numpy())

        # Put it into actor_mem (which is used for training), if the absolute value of the reward is high enough
        if abs(short_mem.reward) > MEMORY_REWARD_THRESH:
            actor_mem.push(short_mem)


# initialize observation tensors
obs_stack = deque(maxlen=INPUT_N_STATES)

next_obs, info = env.reset()
next_obs = torch.tensor(next_obs).to(device)

while len(obs_stack) < INPUT_N_STATES:
    obs_stack.append(next_obs)

next_state_tensor = torch.cat([*obs_stack], dim=0).to(device)

cumulative_reward = 0
def model_infer():
    """
    1. Observe environment
    2. Make a prediction w/ epsilon greedy policy.
    3. Perform the action.
    4. Attempt to train.

    Repeat until the episode ends.
    """
    global step, obs_stack, cumulative_reward, next_obs, next_state_tensor

    done = False
    cumulative_reward = 0
    while not done:
        state_tensor = next_state_tensor.unsqueeze(0)

        actor_model.eval()
        with torch.no_grad():
            out, _ = actor_model.forward(state_tensor)

            multiplot.add_entry('output_0', float(out.clone()[0].tolist()[0]))
            multiplot.add_entry('output_1', float(out.clone()[0].tolist()[1]))
            multiplot.add_entry('output_2', float(out.clone()[0].tolist()[2]))
            multiplot.add_entry('output_3', float(out.clone()[0].tolist()[3]))

        Q, max_a = torch.max(out, dim=1)

        next_obs, reward, terminated, truncated, info = env.step(max_a.cpu().numpy()[0])
        
        multiplot.add_entry('natural_reward', reward)

        cumulative_reward += reward
        multiplot.add_entry('cumulative_reward', cumulative_reward)

        # terminated is if the pole falls. truncated is when the game times out.
        if terminated or truncated:
            next_obs, info = env.reset()
            cumulative_reward = 0 # reset cumulative reward
            done = True # end episode


            # if terminated:
            #     reward = -10 # punishment for losing
            # TERMINATED already is punished in lunar lander

        affect_short_mem(reward)
        
        next_obs = torch.tensor(next_obs).to(device)
        obs_stack.append(next_obs)
        next_state_tensor = torch.cat([*obs_stack], dim=0).to(device)
        
        reward = torch.tensor(np.expand_dims(reward, 0), dtype=torch.float32).to(device)

        mem_block = [state_tensor, max_a, next_state_tensor.unsqueeze(0), reward]

        short_memory.append(mem_block)

        if done: send_short_to_long_mem(len(short_memory))

        try_learning()

        model_adjuster.soft_hard_copy(step, actor_model, pred_model)
        step += 1



def model_train(batch_size):
    """
    This function trains the model using Double-DQN, where the actor_model predicts the next action and then the predictor
    predicts the Q-value of that action for stability reasons.

    Parameters:
        batch_size (int): The amount of samples to include in a minibatch of training.
    
    Returns:
        actor_loss (torch.tensor): Returns the loss of the actor, essentially its error from the target outputs.
    """
    actor_model.train()

    transitions = actor_mem.sample(batch_size)
    mem_batch = Transition(*zip(*transitions))

    # Concatenate mem_batch elements to tensors batches
    state_batch = torch.cat(mem_batch.state, dim=0).to(device)
    action_batch = torch.cat(mem_batch.action, dim=0).to(device)
    next_state_batch = torch.cat(mem_batch.next_state, dim=0).to(device)
    reward_batch = torch.cat(mem_batch.reward, dim=0).to(device) # 64

    # Get the new model output for each state in the batch, including a guess at the next state
    state_values, next_state_guess = actor_model.forward(state_batch, real_actions=action_batch, training=True)
    pred_diff = next_state_batch - next_state_guess
    abs_pred_diff = torch.abs(pred_diff)
    
    diff_from_mean_pred_diff = abs_pred_diff - torch.mean(abs_pred_diff)
    surprisal = torch.sum(diff_from_mean_pred_diff, dim=1)
    scaled_surprisal = (surprisal + SURPRISAL_BIAS) * SURPRISAL_WEIGHT
    multiplot.add_entry("surprisal", (torch.max(scaled_surprisal) - torch.min(scaled_surprisal)).cpu().detach().numpy() * 5)
    
    # Gather the Q-value of the actual actions chosen.
    state_actions = state_values.gather(1, action_batch.unsqueeze(1)) # 64, 1

    with torch.no_grad():
        # Select next action using current model
        actor_next_preds, _ = actor_model.forward(next_state_batch, training=True) # 64, 2
        Q, actor_pred_max_a = torch.max(actor_next_preds, dim=1) # 64
        
        # Predict target Q-value at next_state using the more stable prediction model
        pred_out, _ = pred_model.forward(next_state_batch, training=True) # 64, 2
        next_state_actions = pred_out.gather(1, actor_pred_max_a.unsqueeze(1)) # 64, 1

    # Generate the target output, by adding the reward at each transition, to the Q-value of the next action (predicted reward) * GAMMA, a discount factor.
    target_output = reward_batch.unsqueeze(1) + (next_state_actions * GAMMA)

    # Loss is the difference between the target outputs and the real outputs,
    # plus the difference between the next state and the predicted next state.
    actor_criterion = nn.HuberLoss()
    actor_loss = actor_criterion(state_actions, target_output) + actor_criterion(next_state_guess, next_state_batch)
    actor_optimizer.zero_grad()
    actor_loss.backward()

    # Log gradient norm
    grad_norm = np.sqrt(sum([torch.norm(p.grad)**2 for p in actor_model.parameters()]).detach().cpu())
    multiplot.add_entry('grad_norm', grad_norm)

    # Clip gradients for stability
    torch.nn.utils.clip_grad_value_(actor_model.parameters(), 1)
    actor_optimizer.step()

    return actor_loss



def main():
    global step, env, next_obs, obs_stack, next_state_tensor

    for epoch in tqdm(range(EPOCHS)):
        # Decide whether to display the environment
        if epoch % SNAPSHOT_INTERVAL == 0 and (epoch != 0 or SHOW_FIRST):
            render_mode = "human"
        else:
            render_mode = None

        # Load a new version of the environment with the chosen render_mode
        next_obs, info = env.reset()

        if render_mode != None: env.render()

        # Re-initialize obervations, etc.
        obs_stack = deque(maxlen=INPUT_N_STATES)
        next_obs = torch.tensor(next_obs).to(device)

        while len(obs_stack) < INPUT_N_STATES:
            obs_stack.append(next_obs)

        next_state_tensor = torch.cat([*obs_stack], dim=0).to(device)

        if len(info) > 0: print(info)

        for episode in tqdm(range(EPISODES_PER_EPOCH)):
            model_infer()

        multiplot.plot_all(step)


main()
