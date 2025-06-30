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

# --- Value Iteration ---------------------------------------------------------

def value_iteration(theta=1e-5):
    states = [(x, y) for x in range(1, COLS+1)
                      for y in range(1, ROWS+1)
                      if (x,y) != OBSTACLE]
    # initialize U(s)=0 for non-terminals, or to reward for terminals
    U = {s: (reward(s) if is_terminal(s) else 0.0) for s in states}

    while True:
        delta_max = 0
        U_new = U.copy()
        for s in states:
            if is_terminal(s):
                continue
            # Bellman update
            q_values = []
            for a in ACTIONS:
                q = sum(p * U[s2] for s2, p in transitions(s, a).items())
                q_values.append(q)
            U_new[s] = reward(s) + GAMMA * max(q_values)
            delta_max = max(delta_max, abs(U_new[s] - U[s]))
        U = U_new
        if delta_max < theta:
            break

    # extract greedy policy
    pi = {}
    for s in states:
        if is_terminal(s):
            pi[s] = None
        else:
            best_a = max(ACTIONS,
                          key=lambda a: sum(p * U[s2]
                                            for s2, p in transitions(s, a).items()))
            pi[s] = best_a
    return U, pi

# --- Policy Iteration --------------------------------------------------------

def policy_iteration():
    states = [(x, y) for x in range(1, COLS+1)
                      for y in range(1, ROWS+1)
                      if (x,y) != OBSTACLE]
    # initialize arbitrary policy (e.g. always 'up')
    pi = {s: (None if is_terminal(s) else 'up') for s in states}
    # initialize U(s)=0
    U = {s: 0.0 for s in states}

    is_value_changed = True
    while True:
        # Policy Evaluation (in-place iterative until small change)
        while True:
            delta_max = 0
            for s in states:
                if is_terminal(s):
                    continue
                # evaluate U under fixed pi
                q = sum(p * U[s2] for s2, p in transitions(s, pi[s]).items())
                U_new = reward(s) + GAMMA * q
                delta_max = max(delta_max, abs(U_new - U[s]))
                U[s] = U_new
            if delta_max < 1e-5:
                break

        # Policy Improvement
        policy_stable = True
        for s in states:
            if is_terminal(s):
                continue
            old_action = pi[s]
            # find best action under current U
            best_a = max(ACTIONS,
                          key=lambda a: sum(p * U[s2]
                                            for s2, p in transitions(s, a).items()))
            pi[s] = best_a
            if best_a != old_action:
                policy_stable = False

        if policy_stable:
            break

    return U, pi

# --- Utility to pretty-print tables -----------------------------------------

def make_tables(U, pi):
    """Returns two pandas DataFrames (utilities and policy) indexed by Y=3→1."""
    util_df   = pd.DataFrame(index=[3,2,1], columns=[1,2,3,4], dtype=float)
    policy_df = pd.DataFrame(index=[3,2,1], columns=[1,2,3,4], dtype=object)
    for y in [3,2,1]:
        for x in [1,2,3,4]:
            if (x,y) == OBSTACLE:
                util_df.at[y,x]   = np.nan
                policy_df.at[y,x] = None
            else:
                util_df.at[y,x]   = U.get((x,y), np.nan)
                policy_df.at[y,x] = pi.get((x,y), None)
    return util_df, policy_df

def print_tables(title, util_df, policy_df):
    print(f"\n=== {title} ===")
    print("\nUtilities U(s):")
    print(util_df.round(3).to_string())
    print("\nPolicy π*(s):")
    print(policy_df.fillna('─').to_string())


# --- Main --------------------------------------------------------------------

if __name__ == "__main__":
    # Value Iteration
    U_vi, pi_vi = value_iteration()
    uti_vi, pol_vi = make_tables(U_vi, pi_vi)
    print_tables("Value Iteration", uti_vi, pol_vi)

    # Policy Iteration
    U_pi, pi_pi = policy_iteration()
    uti_pi, pol_pi = make_tables(U_pi, pi_pi)
    print_tables("Policy Iteration", uti_pi, pol_pi)
