# Author: Mahdi Ajami
# LinkedIn: https://www.linkedin.com/in/mahdiajami
# Instagram: https://www.instagram.com/mjc.1400/
# Repository: https://github.com/Jarvis-Mi/Foundations-of-Decision-Making-in-Reinforcement-Learning

import numpy as np
import matplotlib.pyplot as plt
# ------------------   Initialize parameters  ------------------ >

# Setting parameters
gamma = 0.9  # Discount rate 90%
states = ['S1', 'S2'] # Statuses
actions = ['stay', 'move'] # Actions

# Set  Rewards
rewards = {
    ('S1', 'stay'): 2,
    ('S1', 'move'): 5,
    ('S2', 'stay'): 2,
    ('S2', 'move'): 5
}

# Transition possibilities: P(next_state | current_state, action)
transition_probabilities = {
    ('S1', 'stay', 'S1'): 1.0,
    ('S1', 'move', 'S2'): 1.0,
    ('S2', 'stay', 'S2'): 1.0,
    ('S2', 'move', 'S1'): 1.0
}

# Initialize states
values = {'S1': 0, 'S2': 0}
# List to store the values ​​of each state in each iteration
value_history = {'S1': [], 'S2': []}


# ------------------   Value Iteration Algorithm ------------------ >

iterations = 50 # Number of repetitions

for i in range(iterations):
    new_values = values.copy() # Copy previous values
    for state in states:
        state_value = []
        for action in actions:
            action_value = 0
            for next_state in states:
                # Probability of transmission
                prob = transition_probabilities.get((state, action, next_state), 0)
                # Value of action
                action_value += prob * (rewards[(state, action)] + gamma * values[next_state])
            state_value.append(action_value)
        new_values[state] = max(state_value)    # Choose the best course of action
    values = new_values  # Update values
    
# ------------------   Save status values ------------------ >

    value_history['S1'].append(values['S1'])
    value_history['S2'].append(values['S2'])

# ------------------    Draw a chart  ------------------ >
plt.plot(range(1, iterations + 1), value_history['S1'], label="Value of S1", marker="o")
plt.plot(range(1, iterations + 1), value_history['S2'], label="Value of S2", marker="o")
plt.xlabel("Iterations")
plt.ylabel("Value")
plt.title("Bellman Equation Value Updates with Transition Probabilities")
plt.legend()
plt.grid(True)
plt.show()
# ------------------    Display final values ------------------ >

print(f"The final value of S1 is: {values['S1']}")
print(f"The final value of S2 is: {values['S2']}")
