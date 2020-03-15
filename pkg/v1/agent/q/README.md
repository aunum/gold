# Q-learning

An implementation fo the Q-learning algorithm with adaptive learning.

## How it works
In Q-learning the agent stores Q-values (quality values) for each state that it encounters. Q-values are determined by the following equation.

![q-learning](https://wikimedia.org/api/rest_v1/media/math/render/svg/678cb558a9d59c33ef4810c9618baf34a9577686)

Q-learning is an off-policy form of [temporal difference](https://en.wikipedia.org/wiki/Temporal_difference_learning). The Agent simply learns by 
storing a quality value for the state that it encountered and the reward that it recieved for the action taken along with 
the discounted future reward. Taking the future reward into account at each value iteration forms a Markov Chain which will converge to the highest 
reward.

An agent will explore or exploit the Q-values based on the `epsilon` hyperparameter.

The implemented agent also employs adaptive learning by which the `alpha` and `epsilon` hyperparameters are dynamically tuned based on the timestep and an `ada divisor` parameter.

Q-learning doesn't work well in continous environments, the [pkg/v1/env](../env/norm.go) package provides a normalization adapter. One of the adapters is for discretization and can be used to make continuous states discrete.

## Examples
See the [experiments](./experiments) folder for example implementations.

## References
- Wiki: https://en.wikipedia.org/wiki/Q-learning
- TD wiki: https://en.wikipedia.org/wiki/Temporal_difference_learning
- Tutorial: https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/
