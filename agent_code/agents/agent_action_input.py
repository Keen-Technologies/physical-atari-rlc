# Copyright 2026 Keen Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class Pooled(nn.Module):
    def __init__(self, input_size, in_channels, action_channels, base_channels, output_channels, 
                 action_layers):
        super().__init__()
        self.cnns = nn.ModuleList()
        out_channels = base_channels
        self.action_layers = action_layers
        layer = 0
        if input_size > 128: 
            input_size = 256        
        while input_size > 2:
            if action_layers & (1<<layer):
                in_channels += action_channels
            c = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)            
            self.cnns.append(c)
            input_size //= 2
            in_channels = out_channels
            out_channels *= 2
            layer += 1
        c = nn.Conv2d(in_channels, output_channels, kernel_size=2, bias=False)
        self.cnns.append(c)

    def forward(self, x, action_ema):
        if x.shape[2] > 128:
            w = (256 - x.shape[3])//2
            h = (256 - x.shape[2])//2
            x = F.pad(x, (w,w,h,h))
        for i, c in enumerate(self.cnns):
            if self.action_layers & (1<<i):
                expanded_actions = action_ema.view(x.shape[0], -1, 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
                x = torch.cat( (x, expanded_actions), dim=1 )
            x = c(x)
            if c == self.cnns[-2]:
                prefinal = x
            if c != self.cnns[-1]:
                x = F.max_pool2d(x, kernel_size=3, padding=1, stride=2)
                x = F.relu(x)
                
        return x.flatten(start_dim=1), prefinal

class SplitOpt:
    def __init__(self, parameters, base_lr, linear_lr, momentum=0.0):
        self.plist = list(parameters)
        base = self.plist[:-1]
        linear = self.plist[-1:]

        self.base_opt = torch.optim.AdamW( base, lr=base_lr )
        self.linear_opt = torch.optim.SGD( linear, lr=linear_lr, momentum=momentum )

    def zero_grad(self):
        self.base_opt.zero_grad()
        self.linear_opt.zero_grad()

    def step(self):
        self.base_opt.step()
        self.linear_opt.step()


class Agent:
    def __init__(self, seed, num_actions, total_frames, device='cuda', **kwargs):
        torch.manual_seed(seed)
        # set CuBLAS environment variable so matmuls can be deterministic
        os.environ['CUBLAS_WORKSPACE_CONFIG']= ':4096:8'
        # must be combined with os.environ['CUBLAS_WORKSPACE_CONFIG']= ':4096:8' before loading torch
        torch.backends.cudnn.benchmark = False
        _d = str(device)
        _det_env = os.environ.get("ROBOTROLLER_TORCH_DETERMINISTIC", "").strip().lower()
        if _det_env in ("0", "false", "no"):
            use_deterministic = False
        elif _det_env in ("1", "true", "yes"):
            use_deterministic = True
        else:
            # torch.use_deterministic_algorithms(True) on CUDA has caused native crashes on some
            # driver/cuDNN stacks (e.g. around max_pool2d); CPU path is unaffected.
            use_deterministic = not _d.startswith("cuda")
        if use_deterministic:
            torch.use_deterministic_algorithms(True)
        elif _d.startswith("cuda"):
            print(
                "[agent_action_input] Skipping torch.use_deterministic_algorithms on CUDA "
                "(set ROBOTROLLER_TORCH_DETERMINISTIC=1 to enable; may trigger driver bugs on some setups).",
                flush=True,
            )
        print(f'seed:{seed}')
        self.ring_size = 400_000
        self.device = device
        self.total_frames = total_frames

        self.obs_size = 128
        self.obs_channels = 3
        self.base_channels = 64
        self.policy_skip = 4
        self.train_skip = 4
        self.observation_ema_log2 = -3
        self.momentum = 0.9
        self.lr_log2 = -14
        self.lr_linear_log2 = -17
        
        self.explore_log2 = -6
        self.target_model_log2 = 5  # for batch norm, we need to copy the complete model instead of blending a fraction of the weights over
        self.train_batch = 32
        self.train_reps = 1
        self.multisteps = 16
        self.discount = 0.99        # per 60 hz frame, so this is a greater discount than the standard 0.99 at 15 hz
        self.online_samples = 4

        self.action_encoding = 1    # 0 = one-hot, 1 = factored
        self.action_ema_rate = 0.875 # 0.5 # 1-2**(-job) # 0.875    # fraction of the history remaining before blending in each new frame
                                        # 0 = immedaitely take new value, 0.875 = have about half of the previous values after 4 frames
                                        # 1 = don't recognize the new action, which means no learning
        self.action_steps = 4       # number of steps into the future to blend either proposed or actual actions
                                    # before submitting to the model with the observations

        self.action_layers = 2        # bit mask of the CNN layers that will have the action ema appended
        
        # Override parameters from config file
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                print(f"Setting {key} to {value}")
            else:
                print(f"Warning: Unknown parameter '{key}' ignored")

        torch.set_default_device(self.device)
        torch.random.manual_seed( seed )
        self.num_actions = num_actions

        # factored atari inputs
        if self.action_encoding == 1 and num_actions == 18:
            self.all_actions = torch.tensor( 
                [
                    [0,1,0, 0,1,0, 1,0], # 'NOOP': <Action.NOOP: 0>,
                    [0,1,0, 0,1,0, 0,1], # 'FIRE': <Action.FIRE: 1>, 
                    [1,0,0, 0,1,0, 1,0], # 'UP': <Action.UP: 2>, 
                    [0,1,0, 0,0,1, 1,0], # 'RIGHT': <Action.RIGHT: 3>, 
                    [0,1,0, 1,0,0, 1,0], # 'LEFT': <Action.LEFT: 4>, 
                    [0,0,1, 0,1,0, 1,0], # 'DOWN': <Action.DOWN: 5>, 
                    [1,0,0, 0,0,1, 1,0], # 'UPRIGHT': <Action.UPRIGHT: 6>, 
                    [1,0,0, 1,0,0, 1,0], # 'UPLEFT': <Action.UPLEFT: 7>, 
                    [0,0,1, 0,0,1, 1,0], # 'DOWNRIGHT': <Action.DOWNRIGHT: 8>, 
                    [0,0,1, 1,0,0, 1,0], # 'DOWNLEFT': <Action.DOWNLEFT: 9>, 
                    [1,0,0, 0,1,0, 0,1], # 'UPFIRE': <Action.UPFIRE: 10>, 
                    [0,1,0, 0,0,1, 0,1], # 'RIGHTFIRE': <Action.RIGHTFIRE: 11>, 
                    [0,1,0, 1,0,0, 0,1], # 'LEFTFIRE': <Action.LEFTFIRE: 12>, 
                    [0,0,1, 0,1,0, 0,1], # 'DOWNFIRE': <Action.DOWNFIRE: 13>, 
                    [1,0,0, 0,0,1, 0,1], # 'UPRIGHTFIRE': <Action.UPRIGHTFIRE: 14>, 
                    [1,0,0, 1,0,0, 0,1], # 'UPLEFTFIRE': <Action.UPLEFTFIRE: 15>, 
                    [0,0,1, 0,0,1, 0,1], # 'DOWNRIGHTFIRE': <Action.DOWNRIGHTFIRE: 16>, 
                    [0,0,1, 1,0,0, 0,1], # 'DOWNLEFTFIRE': <Action.DOWNLEFTFIRE: 17>}                    
                ]
                )
        else:
            self.all_actions = F.one_hot( torch.arange(self.num_actions) )
        self.all_actions = self.all_actions.float().view(1,num_actions,-1).expand(self.train_batch,num_actions,-1)
        self.action_ema = torch.zeros(self.all_actions.shape[-1])

        self.observation_ema = torch.zeros(self.obs_channels, self.obs_size, self.obs_size)
        # the EMA of one-hot actions up to the observation, not including the action selected for the observation
        nbytes = (
            self.ring_size
            * self.obs_channels
            * self.obs_size
            * self.obs_size
        )

        self.action_ema_ring = torch.zeros(self.ring_size,self.all_actions.shape[-1])
        self.reward_ring = torch.zeros(self.ring_size)
        self.observation_ring = torch.zeros(self.ring_size, self.obs_channels, self.obs_size, self.obs_size, dtype=torch.uint8)
        # model
        self.model = Pooled(self.obs_size, self.obs_channels, self.all_actions.shape[-1], self.base_channels, 1, 
                            self.action_layers)

        self.model.train()
        print(self.model)

        # copy target model over
        self.target_model = copy.deepcopy(self.model)
        self.target_model.eval()

        self.optimizer = SplitOpt(self.model.parameters(), 2**self.lr_log2, 2**self.lr_linear_log2, momentum=self.momentum)
        self.f = 0
        self.selected_action_index = 0
        self.loss = torch.tensor(0.0)

    def observations(self, indexes):
        assert (indexes > self.f - self.ring_size).all()

        obs = self.observation_ring[indexes%self.ring_size].float()
        std, mean = torch.std_mean(obs, dim=(2,3), keepdim=True)
        standardized = (obs - mean) / (std+1e-8) 
        return standardized

    def save_model(self, filepath):
        """Save only the model weights needed for inference."""
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        """Load only the model weights needed for inference."""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))

    def get_state(self):
        """Return complete state dictionary for serialization, including both models."""
        return {
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
        }
    
    def load_state(self, state_dict):
        """Load complete state dictionary, including both models."""
        self.model.load_state_dict(state_dict['model_state_dict'])
        self.target_model.load_state_dict(state_dict['target_model_state_dict'])

    # ignore the end_of_episode signal
    def get_action(self, observation_rgb8, reward, end_of_episode, train=True):
        ring = self.f%self.ring_size

        self.reward_ring[ring] = reward

        # resample the observation into observation_ring
        if self.obs_channels == 1:
            # greyscale
            obs = torch.from_numpy(observation_rgb8).to(self.device).float().mean(dim=2).unsqueeze(dim=0)
        else:
            obs = torch.from_numpy(observation_rgb8).permute(2,0,1).to(self.device).float()
        resampled = F.interpolate( obs.unsqueeze(dim=0), (self.obs_size, self.obs_size), mode='area')[0]
        torch.lerp( self.observation_ema, resampled, 2**self.observation_ema_log2, out=self.observation_ema )
        self.observation_ring[ring] = self.observation_ema.to(dtype=torch.uint8)

        # blend the previous action into the ema buffer
        one_hot = self.all_actions[0,self.selected_action_index]
        torch.lerp( one_hot, self.action_ema, self.action_ema_rate, out=self.action_ema )
        self.action_ema_ring[ring] = self.action_ema

        # possibly select a new action
        if self.f % self.policy_skip == 0:
            if torch.rand(1) < 2**self.explore_log2 and train:
                # random action
                self.selected_action_index = torch.randint( self.num_actions, (1,) )[0].item()
            else:
                with torch.no_grad():
                    policy_observations = self.observations( torch.tensor( [self.f] ) )

                    # expand the current action_ema for blending with all the possible action EMA
                    expanded_action_ema = self.action_ema.view(1,1,-1).expand(1,self.num_actions,-1)
                    
                    # blend in the actions we are evaluating
                    forward_action_ema_rate = self.action_ema_rate ** self.action_steps
                    action_emas = torch.lerp( self.all_actions[0], expanded_action_ema, forward_action_ema_rate )

                    policy_observations = policy_observations.expand(self.num_actions, -1, -1, -1)
                    action_emas = action_emas.squeeze(dim=0)

                    outputs, prefinal = self.target_model( policy_observations, action_emas )
                    self.selected_action_index = outputs.argmax(dim=0).item()
        return self.selected_action_index

    def learn(self, observation_rgb8, reward, end_of_episode, train=True, update_weights=True):
        if train and self.f % self.train_skip == 0 and self.f >= self.multisteps:

            assert not torch.isnan(self.loss).any()

            # blend a fraction into the target model
            if self.target_model_log2 < 0:
                train_list = list(self.model.parameters())
                target_list = list(self.target_model.parameters())
                with torch.no_grad():
                    for i in range(len(train_list)):
                        torch.lerp(target_list[i], train_list[i], 2 ** self.target_model_log2, out=target_list[i])
            elif self.f % 2 ** self.target_model_log2 == 0:
                self.target_model = copy.deepcopy(self.model)
                self.target_model.eval()

            with torch.no_grad():
                # get the state value at the bootstrap point
                low = max(self.multisteps, self.f + 1 - self.ring_size + self.multisteps)
                high = self.f + 1 - self.online_samples
                # Skip training if range is invalid
                if low >= high:
                    self.f += 1
                    return
                bootstrap_indexes = torch.randint(low, high, (self.train_batch,))
                bootstrap_indexes[:self.online_samples] = self.f - torch.arange(self.online_samples)

                bootstrap_observations = self.observations(bootstrap_indexes)

                # load the action EMA from when the observation was seen
                prior_action_emas = self.action_ema_ring[bootstrap_indexes % self.ring_size]

                # expand for all the possible action EMA
                expanded_history = prior_action_emas.view(self.train_batch, 1, -1).expand(self.train_batch,
                                                                                          self.num_actions, -1)

                # blend in the actions we are evaluating
                forward_action_ema_rate = self.action_ema_rate ** self.action_steps
                action_emas = torch.lerp(self.all_actions, expanded_history, forward_action_ema_rate)
                action_emas = action_emas.reshape(self.train_batch * self.num_actions, -1)

                expanded_observation_stacks = bootstrap_observations.unsqueeze(dim=1).expand(self.train_batch,
                                                                                             self.num_actions,
                                                                                             self.obs_channels,
                                                                                             self.obs_size,
                                                                                             self.obs_size)
                expanded_observation_stacks = expanded_observation_stacks.reshape(self.train_batch * self.num_actions,
                                                                                  self.obs_channels, self.obs_size,
                                                                                  self.obs_size)
                # evaluate the model
                all_q, prefinal = self.target_model(expanded_observation_stacks, action_emas)
                all_q = all_q.view(self.train_batch, self.num_actions)

                target_state_values = all_q.amax(dim=1)

                # calculate the discounted return targets
                train_indexes = bootstrap_indexes - self.multisteps
                reward_indexes = train_indexes.unsqueeze(dim=1) + torch.arange(1, self.multisteps + 1).unsqueeze(dim=0)
                rewards = self.reward_ring[reward_indexes.flatten() % self.ring_size].view(self.train_batch,
                                                                                           self.multisteps)
                discount = self.discount ** torch.arange(self.multisteps)
                discounted_rewards = (rewards * discount.unsqueeze(dim=0)).sum(dim=1)
                target_values = discounted_rewards + target_state_values * self.discount ** self.multisteps

                # build the observations + actions for training
                observation_stacks = self.observations(train_indexes)

                # take the action ema from action_steps later, after the action has been applied
                action_ring_indexes = (train_indexes + self.action_steps)
                action_emas = self.action_ema_ring[action_ring_indexes % self.ring_size]

            # evaluate the model with gradients
            for r in range(self.train_reps):
                train_values, prefinal = self.model(observation_stacks, action_emas)
                train_values = train_values.flatten()
                loss_individual = F.mse_loss(train_values, target_values, reduction='none')
                loss = loss_individual.mean()
                self.optimizer.zero_grad()
                loss.backward()
                if update_weights:
                    self.optimizer.step()

        self.f += 1

    
# have the benchmark harness run this agent
if __name__ == '__main__':
    import atari_bench
    import os
    this_module = os.path.splitext(os.path.basename(__file__))[0]
    argv = [ 'atari_bench.py', '--instance_id', '0', this_module, '--instance_dir', 'debug_job', '--game', 'ms_pacman' ]
    atari_bench.main(argv)
