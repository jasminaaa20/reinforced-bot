import numpy as np
import pandas as pd

# --- MDP definition ----------------------------------------------------------

# Grid size
COLS, ROWS = 4, 3

# Obstacle and terminal states
OBSTACLE    = (2, 2)
TERMINALS   = {
    (4, 3): +1.0,
    (4, 2): -1.0
}

# Actions and movement deltas
ACTIONS = ['up', 'down', 'left', 'right']
DELTAS  = {
    'up':    ( 0, +1),
    'down':  ( 0, -1),
    'left':  (-1,  0),
    'right': (+1,  0)
}
# “Perpendicular” slip directions
PERP = {
    'up':    ['left', 'right'],
    'down':  ['left', 'right'],
    'left':  ['up',   'down' ],
    'right': ['up',   'down' ]
}

# Discount and step-reward
GAMMA = 0.9
STEP_REWARD = -0.04
