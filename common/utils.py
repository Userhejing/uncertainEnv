import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np
from typing import Type, Union

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_device():
  return 'cuda' if torch.cuda.is_available() else 'cpu'

def Soft_Update(
    main_net: Type[nn.Module],
    target_net: Type[nn.Module],
    tau: Union[int, float] = 0.99
) -> None:
    """
    Soft updating the parameters of target network.
    target_net = tau*target_net + (1-tau)*main_net.
    Args:
        main_net: the main net in RL.
        target_net: the target_net waiting for soft updating.
        tau: coefficient.
    Returns:
        None, bcz .data is in_place operation.
    """
    with torch.no_grad():
        for param_main, param_target in zip(main_net.parameters(), target_net.parameters()):
            param_target_ = tau*param_target + (1-tau)*param_main
            param_target.data = param_target_  # use .data property is necessary.

def Categorical_action(
    logits : Type[torch.Tensor] = None,
    probs: Type[torch.Tensor] = None,
) -> Type[np.ndarray]:
    """
    sample actions according to categorical distribution.
    Args:
        logits: log probability of each action.
        probs: probability of each action.
    Returns:
        actions. (n,)
    """
    assert (logits is not None) or (probs is not None)
    if logits is not None:
        action_distributions = Categorical(logits = logits)
    else:
        action_distributions = Categorical(probs = probs)
    return action_distributions.sample().numpy()

def Epsilon_greedy_action(
    logits: Type[torch.Tensor] = None,
    probs: Type[torch.Tensor] = None,
    epsilon: float = 0,
) -> Type[np.ndarray]:
    """
    sample actions according to epsilon greedy strategy.
    Args:
        logits: action-state value for each action.
        probs: probability for each action.
        epsilon: just as its name implies.
    Returns:
        actions. (n,)
    """
    assert (logits is not None) or (probs is not None)
    assert 0<=epsilon and epsilon<=1
    rand_ = np.random.rand()
    act_result = logits if logits is not None else probs
    assert len(act_result.shape) == 2
    if rand_ > epsilon:
        actions = torch.argmax(act_result, dim=1).numpy()
    else:
        batch_size = act_result.shape[0]
        act_dim = act_result.shape[1]
        actions = np.random.randint(low=0, high=act_dim, size=(batch_size,))
    return actions

def Continuous_action_noise(
    actions: Type[torch.Tensor],
    sigma: float
) -> Type[torch.Tensor]:
    """
    Add Gaussian noise to the input continuous actions.
    Args:
        actions: continuous actions. two-dimension by default.
        sigma: the standard deviation of noise Gaussion distribution.
    Returns:
        noised_actions.
    """
    if len(actions.shape) == 1:
        actions = actions.unsqueeze(0)
    assert len(actions.shape) == 2
    noise_dist = torch.distributions.normal.Normal(loc=0,scale=sigma)
    noise_samples = noise_dist.sample(sample_shape=actions.shape)
    return actions + noise_samples

def Norm_to_Action(
    actions: Type[torch.Tensor],
    act_high: Type[np.ndarray],
    act_low: Type[np.ndarray]
) -> Type[np.ndarray]:
    """
    rescale the normalized actions (-1, 1) to the original actions (low, high).
    Args:
        actions: the normalized actions. two-dimension by default.
        act_high: the upper bound of original action.
        act_low:  the lower bound of original action.
    Returns:
        original actions (low, high).
    """
    if len(actions.shape) == 1:
        actions = actions.unsqueeze(0)
    assert len(actions.shape) == 2
    act_high, act_low = torch.from_numpy(np.asarray(act_high)).float(),\
                        torch.from_numpy(np.asarray(act_low)).float()
    if len(act_high.shape) == 1:
        act_high = act_high.unsqueeze(0)
        act_low = act_low.unsqueeze(0)
    batch_size = actions.shape[0]
    act_high = torch.cat([act_high for _ in range(batch_size)])
    act_low = torch.cat([act_low for _ in range(batch_size)])
    actions = act_low + (act_high-act_low)/2.0*(actions+1.0)
    actions = torch.clip(actions, min=act_low, max=act_high)
    return actions.numpy()

def Action_to_Norm(
    actions: Type[np.ndarray],
    act_high: Type[np.ndarray],
    act_low: Type[np.ndarray]
) -> Type[torch.Tensor]:
    """
    normalizing the original actions (low, high) to the normalized actions (-1, 1).
    Args:
        actions: original actions (low, high). two-dimension by default.
        act_high: the upper bound of original action.
        act_low:  the lower bound of original action.
    Returns:
        normalized actions (-1, 1).
    """
    actions = torch.from_numpy(actions)
    if len(actions.shape) == 1:
        actions = actions.unsqueeze(0)
    assert len(actions.shape) == 2
    act_high, act_low = torch.from_numpy(np.asarray(act_high)).float(),\
                        torch.from_numpy(np.asarray(act_low)).float()
    if len(act_high.shape) == 1:
        act_high = act_high.unsqueeze(0)
        act_low = act_low.unsqueeze(0)
    batch_size = actions.shape[0]
    act_high = torch.cat([act_high for _ in range(batch_size)])
    act_low = torch.cat([act_low for _ in range(batch_size)])
    actions = 2.0*(actions-act_low)/(act_high-act_low) -1.0
    actions = torch.clip(actions, min=-1, max=1)
    return actions
