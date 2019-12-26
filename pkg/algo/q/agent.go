package q

/*
Q is an implementation of the Q learning equation.

 equation:
 Q(state,action)←(1−α)Q(state,action)+α(reward+γmaxaQ(next state,all actions))

 α (alpha) is the learning rate (0<α≤1) - Just like in supervised learning settings, α is the extent
 to which our Q-values are being updated in every iteration.

 γ (gamma) is the discount factor (0≤γ≤1) - determines how much importance we want to give to future
 rewards. A high value for the discount factor (close to 1) captures the long-term effective award, whereas,
 a discount factor of 0 makes our agent consider only immediate reward, hence making it greedy.
*/

// Agent is a Q-learning agent.
type Agent struct {
}
