# Q-learning

An implementation fo the Q learning algorithm with adaptive learning.

In Q-learning the agent stores Q-values (quality values) for each state that it encounters. Q-values are determined by the following equation.
```
Q(state,action)←(1−α)Q(state,action)+α(reward+γmaxaQ(next state,all actions))
```
An agent will explore or exploit the Q-values based on the `epsilon` hyperparameter.

The implemented agent also employs adaptive learning by which the `alpha` and `epsilon` hyperparameters are dynamically tuned based on the timestep and a `ada divisor`.

See [agent_test.go](agent_test.go) for an implemetation example. It also includes a grid search to find the best hyperparameters.