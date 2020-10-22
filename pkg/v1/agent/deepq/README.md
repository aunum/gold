# Deep Q-learning

Implementation of the DeepQ algorithm with Double Q. 

## How it works

[DeepQ](https://arxiv.org/abs/1312.5602) is an progression on standard [Q-learning](https://en.wikipedia.org/wiki/Q-learning).

![q-learning](https://wikimedia.org/api/rest_v1/media/math/render/svg/678cb558a9d59c33ef4810c9618baf34a9577686)

With DeepQ, rather than storing Q-values in a table, they are aprroximated using neural networks. This allows for more accurate 
Q-value estimates as well as the ability to model continuous states.

DeepQ also includes the notion of experience replay, in which the agent stores the states, actions, and outcomes at every 
step in memory and then randomly samples from them during training. 

[Double-Q](https://arxiv.org/abs/1509.06461) is further implemented in which the target, or expected future rewards, is modeled in a separate network 
having the weights intermittently copied over from the 'online' network making the predictions. This helps learning by 
providing a more stable target to pursue.

## Examples
See the [experiments](./experiments) folder for example implementations.

## Roadmap
- [ ] Prioritized replay
- [ ] Dueling Q
- [ ] Soft updates
- [ ] More environments

## References
- DeepQ paper: https://arxiv.org/abs/1312.5602
- Double Q paper: https://arxiv.org/abs/1509.06461
- Tutorial: https://towardsdatascience.com/deep-q-learning-for-the-cartpole-44d761085c2f
- Tutorial: https://towardsdatascience.com/cartpole-introduction-to-reinforcement-learning-ed0eb5b58288