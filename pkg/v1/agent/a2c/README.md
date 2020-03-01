# Advantage Actor Critic
An implementation of Advantage Actor Critic.

## How it works
Advantage Actor Critic is the synchronous form of A3C. Actor Critic methods are a form of [Policy Gradients](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html) in which the Critic estimates the value function and the Actor updates the policy distribution. The advantage 
tells how much better a specific action is from the distribution of actions.

## Examples
See the [experiments](./experiments) folder for example implementations.

## Roadmap
- [ ] 

## References
- A3C Paper: https://arxiv.org/abs/1602.01783
- Tutorial: https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f
- Release: https://openai.com/blog/baselines-acktr-a2c