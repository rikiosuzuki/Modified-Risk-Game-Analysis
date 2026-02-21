import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

# ═══════════════════════════════════════════════════════════════════════
# PART 1: GAME ENGINE
# ═══════════════════════════════════════════════════════════════════════

def single_roll_probability(a_armies, b_armies):
    """
    Calculate P(A wins roll), P(B wins roll), P(Tie) for a single combat round.
    
    Rules:
    - A rolls 1d6 (always)
    - B rolls 1d4 if 1 army, 2d4 if 2+ armies
    - A wins if A's roll > all of B's rolls → B loses an army
    - B wins if A's roll < at least one of B's rolls → A loses an army
    - Tie if A's roll == max(B's rolls) → no army lost
    """
    if a_armies == 1 or b_armies == 0:
        return 0, 0, 0

    a_wins = b_wins = ties = total = 0

    for a_roll in range(1, 7):
        if b_armies == 1:
            for b_roll in range(1, 5):
                total += 1
                if a_roll > b_roll:
                    a_wins += 1
                elif a_roll < b_roll:
                    b_wins += 1
                else:
                    ties += 1
        else:
            for b1 in range(1, 5):
                for b2 in range(1, 5):
                    total += 1
                    max_b = max(b1, b2)
                    if a_roll > max_b:
                        a_wins += 1
                    elif a_roll < max_b:
                        b_wins += 1
                    else:
                        ties += 1

    return a_wins / total, b_wins / total, ties / total


def build_transition_matrix(m, n):
    """Build the full transition matrix P for a game starting with A=m, B=n."""
    states = []
    state_to_idx = {}

    for a in range(m, 0, -1):
        for b in range(n, -1, -1):
            state_to_idx[(a, b)] = len(states)
            states.append((a, b))

    size = len(states)
    P = np.zeros((size, size))

    for i, (a, b) in enumerate(states):
        if a == 1 or b == 0:
            P[i][i] = 1.0
            continue
        p_a_wins, p_b_wins, p_tie = single_roll_probability(a, b)
        P[i][state_to_idx[(a, b - 1)]] = p_a_wins
        P[i][state_to_idx[(a - 1, b)]] = p_b_wins
        P[i][i] = p_tie

    return P, states, state_to_idx


# ═══════════════════════════════════════════════════════════════════════
# PART 2: ABSORBING MARKOV CHAIN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def analyze_markov_chain(m, n):
    """Full absorbing Markov chain analysis: N = (I-Q)^-1, B = NR, t = N·1"""
    P, states, state_to_idx = build_transition_matrix(m, n)

    transient = []
    absorbing = []
    for i, (a, b) in enumerate(states):
        if a == 1 or b == 0:
            absorbing.append(i)
        else:
            transient.append(i)

    Q = P[np.ix_(transient, transient)]
    R = P[np.ix_(transient, absorbing)]

    I = np.eye(len(transient))
    N = np.linalg.inv(I - Q)
    B = N @ R
    expected_steps = N @ np.ones(len(transient))

    start_idx = transient.index(state_to_idx[(m, n)])

    p_a_wins = sum(
        B[start_idx][j]
        for j, abs_idx in enumerate(absorbing)
        if states[abs_idx][1] == 0
    )

    return {
        'P': P, 'Q': Q, 'R': R, 'N': N, 'B': B,
        'states': states, 'transient': transient,
        'absorbing': absorbing, 'state_to_idx': state_to_idx,
        'p_a_wins': p_a_wins, 'p_b_wins': 1 - p_a_wins,
        'expected_steps': expected_steps[start_idx],
        'all_expected_steps': expected_steps,
    }


# ═══════════════════════════════════════════════════════════════════════
# PART 3: VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════

DARK = {
    'bg': '#0d1117', 'card': '#161b22', 'border': '#30363d',
    'text': '#c9d1d9', 'dim': '#8b949e', 'grid': '#21262d',
    'accent': '#58a6ff', 'green': '#3fb950', 'red': '#f85149',
    'orange': '#d29922', 'purple': '#bc8cff', 'cyan': '#39d2c0',
}

plt.rcParams.update({
    'figure.facecolor': DARK['bg'],
    'axes.facecolor': DARK['card'],
    'axes.edgecolor': DARK['border'],
    'axes.labelcolor': DARK['text'],
    'text.color': DARK['text'],
    'xtick.color': DARK['dim'],
    'ytick.color': DARK['dim'],
    'grid.color': DARK['grid'],
    'legend.facecolor': DARK['card'],
    'legend.edgecolor': DARK['border'],
    'font.family': 'monospace',
})


def plot_transition_heatmap(results, save_path):
    P = results['P']
    states = results['states']
    labels = [f"({a},{b})" for a, b in states]

    fig, ax = plt.subplots(figsize=(14, 12))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'risk_blue', ['#0d1117', '#0d2847', '#1158a8', '#58a6ff', '#79c0ff']
    )
    im = ax.imshow(P, cmap=cmap, aspect='equal', vmin=0, vmax=0.55)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)

    for i in range(len(states)):
        for j in range(len(states)):
            if P[i][j] > 0.001:
                color = DARK['text'] if P[i][j] < 0.4 else DARK['bg']
                ax.text(j, i, f'{P[i][j]:.3f}', ha='center', va='center',
                        fontsize=5.5, color=color, fontweight='bold')

    for i, (a, b) in enumerate(states):
        if a == 1 or b == 0:
            rect = plt.Rectangle((i - 0.5, i - 0.5), 1, 1, linewidth=2,
                                 edgecolor=DARK['green'], facecolor='none', linestyle='--')
            ax.add_patch(rect)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Transition Probability', color=DARK['text'])
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=DARK['dim'])
    ax.set_xlabel('To State (a, b)', fontsize=12)
    ax.set_ylabel('From State (a, b)', fontsize=12)
    ax.set_title('Transition Matrix P — Green borders = Absorbing States',
                 fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_win_probability_surface(save_path, max_m=8, max_n=7):
    m_range = range(2, max_m + 1)
    n_range = range(1, max_n + 1)
    prob_grid = np.zeros((len(list(m_range)), len(list(n_range))))
    steps_grid = np.zeros_like(prob_grid)

    for i, m in enumerate(m_range):
        for j, n in enumerate(n_range):
            res = analyze_markov_chain(m, n)
            prob_grid[i][j] = res['p_a_wins']
            steps_grid[i][j] = res['expected_steps']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    cmap_rg = mcolors.LinearSegmentedColormap.from_list('rg', [DARK['red'], DARK['orange'], DARK['green']])

    im1 = ax1.imshow(prob_grid, cmap=cmap_rg, aspect='auto', vmin=0, vmax=1)
    ax1.set_xticks(range(len(list(n_range))))
    ax1.set_yticks(range(len(list(m_range))))
    ax1.set_xticklabels([str(n) for n in n_range])
    ax1.set_yticklabels([str(m) for m in m_range])
    ax1.set_xlabel("B's Armies (n)", fontsize=12)
    ax1.set_ylabel("A's Armies (m)", fontsize=12)
    ax1.set_title('P(A Wins)', fontsize=14, fontweight='bold')
    for i in range(prob_grid.shape[0]):
        for j in range(prob_grid.shape[1]):
            val = prob_grid[i][j]
            color = DARK['bg'] if 0.35 < val < 0.75 else DARK['text']
            ax1.text(j, i, f'{val:.0%}', ha='center', va='center',
                     fontsize=10, color=color, fontweight='bold')
    ax1.contour(prob_grid, levels=[0.5], colors=['#ffffff'], linewidths=2, linestyles='dashed')
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.85)
    cbar1.set_label('P(A Wins)', color=DARK['text'])
    plt.setp(plt.getp(cbar1.ax.axes, 'yticklabels'), color=DARK['dim'])

    cmap_heat = mcolors.LinearSegmentedColormap.from_list('steps', ['#0d2847', '#1158a8', '#58a6ff', DARK['orange'], DARK['red']])
    im2 = ax2.imshow(steps_grid, cmap=cmap_heat, aspect='auto')
    ax2.set_xticks(range(len(list(n_range))))
    ax2.set_yticks(range(len(list(m_range))))
    ax2.set_xticklabels([str(n) for n in n_range])
    ax2.set_yticklabels([str(m) for m in m_range])
    ax2.set_xlabel("B's Armies (n)", fontsize=12)
    ax2.set_ylabel("A's Armies (m)", fontsize=12)
    ax2.set_title('Expected Game Length', fontsize=14, fontweight='bold')
    for i in range(steps_grid.shape[0]):
        for j in range(steps_grid.shape[1]):
            ax2.text(j, i, f'{steps_grid[i][j]:.1f}', ha='center', va='center',
                     fontsize=10, color=DARK['text'], fontweight='bold')
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.85)
    cbar2.set_label('Expected Steps', color=DARK['text'])
    plt.setp(plt.getp(cbar2.ax.axes, 'yticklabels'), color=DARK['dim'])

    fig.suptitle('Win Probability & Game Length Across Configurations',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_win_prob_lines(save_path, max_m=8, max_n=6):
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [DARK['accent'], DARK['green'], DARK['orange'], DARK['purple'], DARK['cyan'], DARK['red']]
    for idx, n in enumerate(range(1, max_n + 1)):
        m_vals, p_vals = [], []
        for m in range(2, max_m + 1):
            res = analyze_markov_chain(m, n)
            m_vals.append(m)
            p_vals.append(res['p_a_wins'] * 100)
        ax.plot(m_vals, p_vals, 'o-', color=colors[idx], linewidth=2.5, markersize=8,
                markeredgecolor=DARK['bg'], markeredgewidth=1.5, label=f'B = {n}', zorder=3)
    ax.axhline(y=50, color=DARK['dim'], linestyle='--', linewidth=1.5, alpha=0.7, label='50%')
    ax.axhspan(50, 100, alpha=0.05, color=DARK['green'])
    ax.axhspan(0, 50, alpha=0.05, color=DARK['red'])
    ax.set_xlabel("A's Armies (m)", fontsize=13)
    ax.set_ylabel("P(A Wins) %", fontsize=13)
    ax.set_title("Win Probability vs A's Army Size", fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_dice_breakdown(save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (label, b_armies) in zip(axes, [("B has 1 army (1d6 vs 1d4)", 1), ("B has 2+ armies (1d6 vs 2d4)", 2)]):
        p_a, p_b, p_t = single_roll_probability(3, b_armies)
        bars = ax.bar(['A Wins\nRoll', 'Tie', 'B Wins\nRoll'], [p_a, p_t, p_b],
                       color=[DARK['green'], DARK['orange'], DARK['red']], width=0.6,
                       edgecolor=DARK['border'], linewidth=1.5)
        for bar, val in zip(bars, [p_a, p_t, p_b]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.1%}', ha='center', fontsize=13, fontweight='bold', color=DARK['text'])
        ax.set_ylim(0, 0.65)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
    fig.suptitle('Single Roll Probability Breakdown', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_expected_steps_bar(save_path, max_m=8, max_n=5):
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [DARK['accent'], DARK['green'], DARK['orange'], DARK['purple'], DARK['cyan']]
    m_range = list(range(2, max_m + 1))
    x = np.arange(len(m_range))
    bar_width = 0.15
    for idx, n in enumerate(range(1, max_n + 1)):
        steps = [analyze_markov_chain(m, n)['expected_steps'] for m in m_range]
        offset = (idx - max_n / 2 + 0.5) * bar_width
        ax.bar(x + offset, steps, bar_width, color=colors[idx], label=f'B = {n}',
               edgecolor=DARK['bg'], linewidth=0.5, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f'A={m}' for m in m_range])
    ax.set_xlabel("A's Armies", fontsize=13)
    ax.set_ylabel("Expected Steps", fontsize=13)
    ax.set_title("Expected Game Length by Configuration", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_fundamental_matrix(results, save_path):
    N = results['N']
    states = results['states']
    transient = results['transient']
    t_labels = [f"({states[i][0]},{states[i][1]})" for i in transient]

    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = mcolors.LinearSegmentedColormap.from_list('purple', ['#0d1117', '#1a1e3a', '#2d1f6e', '#6e40aa', DARK['purple']])
    im = ax.imshow(N, cmap=cmap, aspect='equal')
    ax.set_xticks(range(len(t_labels)))
    ax.set_yticks(range(len(t_labels)))
    ax.set_xticklabels(t_labels, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(t_labels, fontsize=7)
    for i in range(N.shape[0]):
        for j in range(N.shape[1]):
            if N[i][j] > 0.05:
                color = DARK['text'] if N[i][j] < np.max(N) * 0.6 else DARK['bg']
                ax.text(j, i, f'{N[i][j]:.2f}', ha='center', va='center', fontsize=5.5, color=color)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Expected Visits', color=DARK['text'])
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=DARK['dim'])
    ax.set_xlabel('To Transient State', fontsize=12)
    ax.set_ylabel('From Transient State', fontsize=12)
    ax.set_title('Fundamental Matrix N = (I − Q)⁻¹', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    M, N_VAL = 5, 4
    os.makedirs('plots', exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Modified Risk Game — Markov Chain Analysis")
    print(f"  A = {M} armies, B = {N_VAL} armies")
    print(f"{'='*60}\n")

    results = analyze_markov_chain(M, N_VAL)
    print(f"  Matrix size:      {results['P'].shape[0]}×{results['P'].shape[0]}")
    print(f"  Transient states: {len(results['transient'])}")
    print(f"  Absorbing states: {len(results['absorbing'])}")
    print(f"  P(A wins):        {results['p_a_wins']:.4f} ({results['p_a_wins']:.1%})")
    print(f"  P(B wins):        {results['p_b_wins']:.4f} ({results['p_b_wins']:.1%})")
    print(f"  Expected steps:   {results['expected_steps']:.2f}")
    print(f"\n  Generating plots...\n")

    plot_transition_heatmap(results, 'plots/plot1_transition_heatmap.png')
    print("  ✓ Transition Matrix Heatmap")

    plot_win_probability_surface('plots/plot2_win_prob_surface.png')
    print("  ✓ Win Probability & Expected Steps Surfaces")

    plot_win_prob_lines('plots/plot3_win_prob_lines.png')
    print("  ✓ Win Probability Line Chart")

    plot_dice_breakdown('plots/plot4_dice_breakdown.png')
    print("  ✓ Dice Probability Breakdown")

    plot_expected_steps_bar('plots/plot5_expected_steps.png')
    print("  ✓ Expected Steps Bar Chart")

    plot_fundamental_matrix(results, 'plots/plot6_fundamental_matrix.png')
    print("  ✓ Fundamental Matrix Heatmap")

    print(f"\n{'='*60}")
    print(f"  All plots saved to plots/ directory")
    print(f"{'='*60}\n")
