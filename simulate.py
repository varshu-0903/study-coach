"""Simulation experiment: adaptive (BKT + prerequisites) vs random question order.
Run: python simulate.py  -> prints results and saves sim_results.png
"""
import random, statistics
import numpy as np

P_INIT, P_LEARN, P_GUESS, P_SLIP = 0.2, 0.15, 0.25, 0.1  # same BKT params as the web app


def bkt(p, correct):
    post = p * (1 - P_SLIP) / (p * (1 - P_SLIP) + (1 - p) * P_GUESS) if correct \
        else p * P_SLIP / (p * P_SLIP + (1 - p) * (1 - P_GUESS))
    return post + (1 - post) * P_LEARN


def random_dag(rng, n):
    return [rng.sample(range(i), min(i, rng.choice([0, 1, 1, 2]))) if i else [] for i in range(n)]


def pick_adaptive(belief, prereqs, last):
    n = len(belief)
    if last and not last[1] and prereqs[last[0]]:  # failed -> diagnose weakest prerequisite
        w = min(prereqs[last[0]], key=lambda j: belief[j])
        if belief[w] < 0.6:
            return w
    todo = [i for i in range(n) if belief[i] < 0.9] or [min(range(n), key=lambda i: belief[i])]
    ready = [i for i in todo if all(belief[j] >= 0.6 for j in prereqs[i])] or todo
    return min(ready, key=lambda i: abs(belief[i] - 0.5) + random.random() * 1e-3)


def run(policy, seed, n=12, cap=600):
    rng = random.Random(seed)
    prereqs = random_dag(rng, n)
    known = [rng.random() < 0.2 for _ in range(n)]
    belief, last = [P_INIT] * n, None
    for step in range(1, cap + 1):
        if all(known):
            return step - 1
        i = pick_adaptive(belief, prereqs, last) if policy == "adaptive" else rng.randrange(n)
        correct = rng.random() > P_SLIP if known[i] else rng.random() < P_GUESS
        belief[i] = bkt(belief[i], correct)
        last = (i, correct)
        if not known[i]:  # learning is much faster when prerequisites are known
            ready = all(known[j] for j in prereqs[i])
            known[i] = rng.random() < (P_LEARN * 1.5 if ready else 0.02)
    return cap


if __name__ == "__main__":
    N = 500
    a = [run("adaptive", s) for s in range(N)]
    r = [run("random", s) for s in range(N)]
    ma, mr = statistics.mean(a), statistics.mean(r)
    print(f"Students simulated: {N} (random concept graphs, 12 concepts each)")
    print(f"Random order : {mr:.1f} questions to mastery")
    print(f"Adaptive     : {ma:.1f} questions to mastery")
    print(f"Adaptive needs {100 * (mr - ma) / mr:.1f}% fewer questions")
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4))
        plt.hist(r, bins=30, alpha=.6, label=f"Random (mean {mr:.0f})")
        plt.hist(a, bins=30, alpha=.6, label=f"Adaptive (mean {ma:.0f})")
        plt.xlabel("Questions until all concepts mastered"); plt.ylabel("Simulated students")
        plt.title("Adaptive vs random question order"); plt.legend(); plt.tight_layout()
        plt.savefig("sim_results.png", dpi=160)
        print("Saved sim_results.png")
    except ImportError:
        pass
