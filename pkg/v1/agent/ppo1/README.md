# Proximal Policy Optimization

__In Progress__ ⚠️ blocked on https://github.com/gorgonia/gorgonia/issues/373

Implementation of the Proximal Policy Optimization algorithm.

## How it works
PPO is an on-policy method that aims to solve the step size issue with policy gradients. Typically policy gradient 
algorithms are very sensitive to step size, too large a step and the agent can fall into an unrecoverable state, to small a 
size and the agent takes a very long time to train. PPO solves this issue by ensuring that an agents policy never deviates too far 
from the previous policy.

![eq](https://miro.medium.com/max/1476/0*S949lemw0fEDVPJE)   
A ratio is taken of the old policy to the new policy and the delta is clipped to ensure policy changes remain within a bounds.

## Examples
See the [experiments](./experiments) folder for example implementations.

## Roadmap
- [ ] waiting on bug https://github.com/gorgonia/gorgonia/issues/373

## References
- Release: https://openai.com/blog/openai-baselines-ppo/
- Paper: https://arxiv.org/pdf/1707.06347.pdf
- Tutorial: https://spinningup.openai.com/en/latest/algorithms/ppo.html
- Tutorial: https://towardsdatascience.com/proximal-policy-optimization-tutorial-part-1-actor-critic-method-d53f9afffbf6