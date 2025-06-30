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

# --- Helper functions --------------------------------------------------------

def in_grid(s):
    """Return True if s=(x,y) is inside grid and not the obstacle."""
    x, y = s
    if x < 1 or x > COLS or y < 1 or y > ROWS:
        return False
    if s == OBSTACLE:
        return False
    return True

def is_terminal(s):
    return s in TERMINALS

def reward(s):
    """Immediate reward of entering state s."""
    if s in TERMINALS:
        return TERMINALS[s]
    return STEP_REWARD

def move(s, action):
    """Deterministic move (bump into wall stays in place)."""
    dx, dy = DELTAS[action]
    s2 = (s[0] + dx, s[1] + dy)
    return s2 if in_grid(s2) else s

def transitions(s, a):
    """
    Returns dict of {s': probability} under action a in state s,
    with 0.8 intended and 0.1 each for the two perpendicular slips.
    """
    if is_terminal(s):
        return {s: 1.0}
    probs = {}
    # intended
    for act, p in [(a, 0.8)] + [(slip, 0.1) for slip in PERP[a]]:
        s2 = move(s, act)
        probs[s2] = probs.get(s2, 0) + p
    return probs
