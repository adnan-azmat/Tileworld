# Team 7 (Agents) Code for AI6125 Group Project

## Baseline

Our project is based on the promising Plan C approach from .

We found the [Tileworld-MAS](https://github.com/RoyalSkye/Tileworld-MAS) approach (specifically, plan C) to be promising, and therefore chose it as our baseline.

## Contributions

Our project introduces the following enhancements to the Tileworld-MAS approach:

### Expansion of Agents

We expanded the number of agents from 2 to 5, adding an additional team of 2 agents and 1 replicated agent. This increased the overall reward notably.

### Improved Fuel Finding Strategy

During experiments we observed that as we increased the size of the grid, there were iterations with very small or zero rewards. On closer inspection, it was deduced that the reason for low rewards is that the agents cannot find the fuel stations in some cases. So, we focused on maximising the probability of the fuel station being found.

To address issues with agents getting stuck or dying due to fuel exhaustion, particularly in larger grids, we improved the fuel finding strategy. The changes are as follows:

- The two agent teams (each with two agents) are assigned zones based on their spawn location.
- The pair of agents closer to the center of the grid meet at the center, then move in an 's' shape towards the top and bottom, respectively.
- The other two agents (the team) closer to the corners of the grid move to their nearest corners and explore these areas in an 's' shape.

These changes didn't significantly improve the reward on smaller grids, where fuel station finding was mostly successful. However, on larger grids, we observed significant improvements in rewards earned with the improved fuel finding strategy.

## Results

We ran 10 experiments on each of the environment summarised in the table below.

|            |  Env1   |  Env2   |
| :--------: | :-----: | :-----: |
|    Size    | 50 * 50 | 80 * 80 |
|    *μ*     |   0.2   |    2    |
|    *σ*     |  0.05   |   0.5   |
|  Lifetime  |   100   |   30    |
| Fuel-level |   500   |   500   |
|   Steps    |  5000   |  5000   |

<p>Table 1: Environment configurations</p>

The results obtained by different approaches are as follows:

| Agents plan                                           | Env Setting |     Reward    |
| :--------:                                           | :---------: | :-----------: |
| Baseline (2 agents)                                  |    Env1     |     320.3     |
| More agents (5 agents)                               |    Env1     |     **522.3**     |
| 5 agents w/ Improved Fuel Station Finding Strategy   |    Env1     |     521.8     |

<p>Table 2: Results on Env1</p>


| Agents plan                                           | Env Setting |     Reward    |
| :--------:                                           | :---------: | :-----------: |
| Baseline (2 agents)                                  |    Env2     |     345.5     |
| More agents (5 agents)                               |    Env2     |     699     |
| 5 agents w/ Improved Fuel Station Finding Strategy   |    Env2     |     **791.6**     |

<p>Table 3: Results on Env2</p>

The nn_approach directory has the Neural Network Approach for the Tileworld environment which is pending due to few bugs and resource constraints.
