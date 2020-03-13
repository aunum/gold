# Hindsight Experience Replay

Hindsight experience replay allows an agent to learn in environments with sparse rewards
and multiple goals.

## How it works
[HER](https://arxiv.org/pdf/1707.01495.pdf) utilizes [UVFAs](https://deepmind.com/research/publications/universal-value-function-approximators) and works by augmenting experience replays with additional goals. The intuition being there is valuable 
information to be learned even when the end goal is not reached e.g. if I miss a shot in basketball I can 
still reason that had the hoop been slightly moved I would have made it.

HER is a sort of intrisic [ciriculum learning](https://towardsdatascience.com/how-to-improve-your-network-performance-by-using-curriculum-learning-3471705efab4) 
in which the agent is able to learn from smaller goals before reaching the larger ones.

## Examples
See the [experiments](./experiments) folder for example implementations.

## Roadmap
- [ ] n>15 on bitflip
- [ ] More hindsight types
- [ ] More environments (push-drag)

## References
- Paper: https://arxiv.org/pdf/1707.01495.pdf
- UVFAs: https://deepmind.com/research/publications/universal-value-function-approximators
- Tutorial: https://deeprobotics.wordpress.com/2018/03/07/bitflipper-herdqn 