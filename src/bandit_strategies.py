import math
import numpy as np

def ucb1(t, model_counts, model_rewards):
    ucb_values = np.zeros(len(model_counts))
    total_counts = np.sum(model_counts)

    for i in range(len(model_counts)):
        if model_counts[i] == 0:
            ucb_values[i] = math.inf
        else:
            exploration = math.sqrt((2 * math.log(total_counts)) / model_counts[i])
            ucb_values[i] = model_rewards[i] / model_counts[i] + exploration

    return np.argmax(ucb_values)
