import argparse
import math
import sys
from collections import deque

def expo_dist(lam, x):
    value = lam * math.exp(-lam * x)
    return max(value, 1e-10)  # ελενχουμε οτι το  ότι το αποτέλεσμα δεν είναι ποτέ μηδέν

def cost_fun(i, j, n, Gama):
    if j > i:
        return Gama * (j - i) * math.log(n)
    return 0

def viterbi_algo(n, T, message_intervals, s=2, Gama=1):
    k = int(1 + math.log(T, 2) + math.log(1 / min(message_intervals), 2))
    g = T / n
    lambdas = [s**i / g for i in range(k)]
    C = [[float('inf')] * k for _ in range(n + 1)]
    C[0][0] = 0
    path = [[0] * k for _ in range(n + 1)]

    for t in range(1, n + 1):
        for j in range(k):
            min_cost = float('inf')
            best_prev_state = 0
            for l in range(k):
                transition_cost = C[t-1][l] + cost_fun(l, j, n, Gama)
                if transition_cost < min_cost:
                    min_cost = transition_cost
                    best_prev_state = l
            dist_value = expo_dist(lambdas[j], message_intervals[t-1])
            C[t][j] = -math.log(max(dist_value, 1e-10)) + min_cost  # ελέγχος για την αποφυγή σφάλματος μαθηματικού τομέα
            path[t][j] = best_prev_state
    # Αναζήτηση της βέλτιστης διαδρομής
    optimal_path = deque()
    last_state = C[n].index(min(C[n]))
    optimal_path.appendleft(last_state)

    for t in range(n, 0, -1):
        last_state = path[t][last_state]
        optimal_path.appendleft(last_state)

    return list(optimal_path)

def parse_times(file_path):
    with open(file_path, 'r') as file:
        # Μορφοποιηση .txt file 
        times = file.readline().strip().split()
        times = [float(time) for time in times]  # μετατροπη σε float 
    return times


def format_output(states, times):
    start = 0.0
    current_state = states[0]
    results = []
    for i in range(1, len(times)):
        if states[i] != current_state:
            results.append(f"{current_state} [{start} {times[i - 1]})")
            current_state = states[i]
            start = times[i - 1]
    results.append(f"{current_state} [{start} {times[-1]})")  #  Τελευταίο διάστημα
    return results

def main():
    parser = argparse.ArgumentParser(description='Process the transmission times and compute the optimal state sequence using specified algorithm.')
    parser.add_argument('algorithm', choices=['viterbi', 'trellis'], help='Algorithm to use (viterbi or trellis)')
    parser.add_argument('file', type=str, help='File containing the emission times of the messages')
    parser.add_argument('-s', type=float, default=2, help='Scaling factor s for the exponential distribution')
    parser.add_argument('-g', '--gamma', type=float, default=1, help='Gamma value for the transition cost')
    parser.add_argument('-d', action='store_true', help='Print diagnostic messages')
    args = parser.parse_args()

    if args.d:
        print(f"Running {args.algorithm} algorithm with parameters s={args.s}, gamma={args.gamma}")

    times = parse_times(args.file)
    message_intervals = [times[i] - times[i - 1] for i in range(1, len(times))]
    n = len(message_intervals)
    T = times[-1] - times[0]  # Συνολικός χρόνος από το πρώτο έως το τελευταίο μήνυμα

    if args.algorithm == 'viterbi':
        optimal_states = viterbi_algo(n, T, message_intervals, args.s, args.gamma)
        output = format_output(optimal_states, times)
    elif args.algorithm == 'trellis':
        # υλοποίηση του αλγορίθμου trellis/Bellman-Ford
        output = ["O Trellis αλγόριθμος δεν έχει υλοποιηθεί ακόμη"]

    for line in output:
        print(line)

if __name__ == "__main__":
    main()
