import numpy as np
import ptan

if __name__ == "__main__":
    q_vals = np.array([[1, 2, 3], [1, -1, 0]])
    print("q_vals", q_vals)

    selector = ptan.actions.ArgmaxActionSelector()
    print("arg_max actions", selector(q_vals))

    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=0)
    print("epsilon 0 actions", selector(q_vals))
    
    selector.epsilon = 1.0
    print("epsilon 1 actions", selector(q_vals))

    selector.epsilon = 0.5
    print("epsilon 0.5 actions", selector(q_vals))

    selector = ptan.actions.ProbabilityActionSelector()
    for _ in range(10):
        acts = selector(np.array([
            [0.1, 0.8, 0.1],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0]
        ]))
        print(acts)