# Natural Evolution Strategies
An implementation of Natural Evolution Strategies for black box optimization.

## How it works
[Natural Evolution Strategies](http://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf) optimizes a black box function parameterized with a set of weights by mutating a baseline 
set of weights with noise for each member of the population, then biasing the noise provided to the rewards achieved
and summing that accross the entire population. The baseline set of weights is then updated at each generation based on the 
[Natural Gradient](https://towardsdatascience.com/its-only-natural-an-excessively-deep-dive-into-natural-gradient-optimization-75d464b89dbb) of the 
entire population.

![eq](https://wikimedia.org/api/rest_v1/media/math/render/svg/61af6a537a386b327c5c1d88b82128ceabb08d6d)
## Examples
See the [experiments](./experiments) folder for example implementations.

## Roadmap
- [ ] More enviornments
- [ ] K8s
- [ ] Support multiple weights.

## References
- Paper - http://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf
- Wiki - https://en.wikipedia.org/wiki/Natural_evolution_strategy
- Karpathy sample code - https://gist.github.com/karpathy/77fbb6a8dac5395f1b73e7a89300318d
- A Visual Guide to Evolution Strategies - http://blog.otoro.net/2017/10/29/visual-evolution-strategies
- OpenAI Release - https://openai.com/blog/evolution-strategies
- Lilian Weng Overview - https://lilianweng.github.io/lil-log/2019/09/05/evolution-strategies.html