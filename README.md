# go-rl

Reinforcement learning in Go!

## Getting Started

All of the agent implementations can be found in `pkg/v1/agents` each agent has an experiment folder providing demos across various environments.

## Agents

- [x] Q
- [ ] A2C
- [ ] ACER
- [ ] ACKTR
- [ ] DDPG
- [x] DQN
- [ ] GAIL
- [ ] HER
- [!] PPO1
- [ ] PPO2
- [ ] SAC
- [ ] TD3
- [ ] TRPO

## Inspiration
- OpenAI Baselines https://github.com/openai/baselines
- The Gorgonia Project https://github.com/gorgonia
- RL Overview by Lilian Weng https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html
- Open Endedness https://www.oreilly.com/radar/open-endedness-the-last-grand-challenge-youve-never-heard-of/
- AI-GAs http://www.evolvingai.org/files/1905.10985.pdf 
- The Bitter Lesson http://incompleteideas.net/IncIdeas/BitterLesson.html


## Ideas
- Neural Logic - paper: https://arxiv.org/pdf/1904.10729.pdf
- Imagination-Augmented Agents - paper: https://arxiv.org/pdf/1707.06203.pdf code: https://github.com/clvrai/i2a-tf release: https://deepmind.com/blog/article/agents-imagine-and-plan
- World Models - paper: https://github.com/clvrai/i2a-tf code: https://github.com/hardmaru/WorldModelsExperiments
- HER - paper: https://arxiv.org/abs/1707.01495 tutorial: 
- Neuroevolution - https://towardsdatascience.com/deep-neuroevolution-genetic-algorithms-are-a-competitive-alternative-for-training-deep-neural-822bfe3291f5
    - https://towardsdatascience.com/reinforcement-learning-without-gradients-evolving-agents-using-genetic-algorithms-8685817d84f
    - https://eng.uber.com/deep-neuroevolution/
    - https://www.oreilly.com/radar/neuroevolution-a-different-kind-of-deep-learning/
- Population based policy gradient - https://designrl.github.io/  https://papers.nips.cc/paper/7785-evolved-policy-gradients.pdf
- NEAT & HyperNEAT - http://blog.otoro.net/2016/05/07/backprop-neat/
- Compositional pattern-producing networks - https://towardsdatascience.com/understanding-compositional-pattern-producing-networks-810f6bef1b88
- Multi-agent - https://arxiv.org/abs/1911.10635
- Novelty search - https://eplex.cs.ucf.edu/papers/lehman_ecj11.pdf
- POET - https://eng.uber.com/poet-open-ended-deep-learning/
- Quality Diversity - https://www.frontiersin.org/articles/10.3389/frobt.2016.00040/full
- Minimal criterion coevolution - http://eplex.cs.ucf.edu/papers/brant_gecco17.pdf
- Environment / Ciriculum generation- https://dl.acm.org/doi/abs/10.1145/3205455.3205517
    - Procedural content generation - https://arxiv.org/abs/1911.13071
    - Progressive PGC - https://arxiv.org/abs/1806.10729
- Neuromodulation - http://www.evolvingai.org/miconi-t-rawal-clune-stanley-2019-backpropamine-training-self
- Coevolutionary Temporal Difference Learning - http://www.cs.put.poznan.pl/mszubert/pub/mscthesis.pdf
- Automatic goal generation - https://arxiv.org/pdf/1705.06366.pdf
- Graph neural networks - https://arxiv.org/abs/1810.09202
- Dynamic HER - https://openreview.net/pdf?id=Byf5-30qFX
- Energy based HER - https://arxiv.org/pdf/1810.01363.pdf