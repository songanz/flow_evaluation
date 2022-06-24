import torch as tr
from torch import nn, Tensor
from torch.nn import Module
from typing import Dict, List
import os


class ModelLoadError(Exception):
    """load model error"""


class FCNetTorchPolicy(Module):
    def __init__(self, input_state_size: int = 334,  # 334: SSM trained policy input shape. Just for debug
                 action_size: int = 1,  # 1: SSM trained policy action shape. Just for debug
                 depth_size: List[int] = (400, 200, 100),
                 alpha: float = 0.01):
        super(FCNetTorchPolicy, self).__init__()
        self.input_state_size = input_state_size
        self.action_size = action_size
        self.depth_size = depth_size
        self.alpha = alpha

        layers_dims = [self.input_state_size] + list(self.depth_size) + [self.action_size]
        dims_pairs = list(zip(layers_dims[:-1], layers_dims[1:]))

        act = self.activation_fn()

        # build up nn layers
        layers: List[Module] = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)

            # if not the last layer, add relu function
            if ind < len(dims_pairs) - 1:
                layers.append(act)
            else:
                continue

        self.q_net = nn.Sequential(*layers)

    def activation_fn(self):
        return nn.LeakyReLU(self.alpha)  # for default, use leaky relu

    def forward(self, x: Tensor) -> Tensor:
        return self.q_net(x)

    def load_model_from_path(self, path: str) -> None:
        ckpt = tr.load(path)
        self._load_pretrain(ckpt["state_dict"])

    def _load_pretrain(self, pretrain_dict: Dict[str, tr.Tensor]) -> None:
        state_dict = self.q_net.state_dict()
        for key in pretrain_dict.keys():
            # todo remove this after training onw model
            if 'mlp.' in key:
                new_key = key.split('mlp.')[-1]
            else:
                new_key = key

            if new_key in state_dict and (pretrain_dict[key].size() == state_dict[new_key].size()):
                value = pretrain_dict[key]
                if not isinstance(value, tr.Tensor):
                    value = value.data
                state_dict[new_key] = value
            else:
                raise ModelLoadError(f'fail to load {key}')

        self.q_net.load_state_dict(state_dict)


# for debug
if __name__ == '__main__':
    pate_model = os.path.join(os.getcwd(), "saved_agents/baseline_2022_02_22_ctx_leaders_pytorch/agent_70/saved_model")
    policy = FCNetTorchPolicy()
    policy.load_model_from_path(pate_model)
    print(policy.q_net)
