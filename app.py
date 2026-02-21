import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ═══════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Modified Risk — Markov Chain Analysis",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark theme CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #58a6ff, #39d2c0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #8b949e;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    .stat-card {
        background: linear-gradient(145deg, #161b22, #0d1117);
        border: 1px solid #30363d;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        font-family: 'Courier New', monospace;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        margin-top: 30px;
        margin-bottom: 5px;
    }
    .math-block {
        background: #161b22;
        border: 1px solid #30363d;
        border-left: 3px solid #58a6ff;
        border-radius: 8px;
        padding: 16px 20px;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        margin: 10px 0;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #161b22, #0d1117);
        border: 1px solid #30363d;
        border-radius: 14px;
        padding: 15px 20px;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# CORE ENGINE (same functions from the analysis)
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data
def single_roll_probability(a_armies, b_armies):
    """Calculate P(A wins roll), P(B wins roll), P(Tie) for a single combat round."""
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


@st.cache_data
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


@st.cache_data
def analyze_markov_chain(m, n):
    """Full absorbing Markov chain analysis."""
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
# MATPLOTLIB SETUP
# ═══════════════════════════════════════════════════════════════════════

DARK = {
    'bg': '#0d1117', 'card': '#161b22', 'border': '#30363d',
    'text': '#c9d1d9', 'dim': '#8b949e', 'grid': '#21262d',
    'accent': '#58a6ff', 'green': '#3fb950', 'red': '#f85149',
    'orange': '#d29922', 'purple': '#bc8cff', 'cyan': '#39d2c0',
}

def apply_dark_style():
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

apply_dark_style()


# ═══════════════════════════════════════════════════════════════════════
# PLOTTING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def plot_transition_heatmap(results):
    P = results['P']
    states = results['states']
    labels = [f"({a},{b})" for a, b in states]

    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'risk_blue', ['#0d1117', '#0d2847', '#1158a8', '#58a6ff', '#79c0ff']
    )
    im = ax.imshow(P, cmap=cmap, aspect='equal', vmin=0, vmax=0.55)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)

    for i in range(len(states)):
        for j in range(len(states)):
            if P[i][j] > 0.001:
                color = DARK['text'] if P[i][j] < 0.4 else DARK['bg']
                ax.text(j, i, f'{P[i][j]:.3f}', ha='center', va='center',
                        fontsize=5, color=color, fontweight='bold')

    for i, (a, b) in enumerate(states):
        if a == 1 or b == 0:
            rect = plt.Rectangle((i - 0.5, i - 0.5), 1, 1, linewidth=2,
                                 edgecolor=DARK['green'], facecolor='none',
                                 linestyle='--')
            ax.add_patch(rect)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Transition Probability', color=DARK['text'])
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=DARK['dim'])

    ax.set_xlabel('To State (a, b)', fontsize=11)
    ax.set_ylabel('From State (a, b)', fontsize=11)
    ax.set_title('Transition Matrix P\nGreen borders = Absorbing States',
                 fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    return fig


def plot_win_probability_surface(max_m=8, max_n=7):
    m_range = range(2, max_m + 1)
    n_range = range(1, max_n + 1)

    prob_grid = np.zeros((len(list(m_range)), len(list(n_range))))
    steps_grid = np.zeros_like(prob_grid)

    for i, m in enumerate(m_range):
        for j, n in enumerate(n_range):
            res = analyze_markov_chain(m, n)
            prob_grid[i][j] = res['p_a_wins']
            steps_grid[i][j] = res['expected_steps']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    cmap_rg = mcolors.LinearSegmentedColormap.from_list(
        'rg', [DARK['red'], DARK['orange'], DARK['green']]
    )
    im1 = ax1.imshow(prob_grid, cmap=cmap_rg, aspect='auto', vmin=0, vmax=1)
    ax1.set_xticks(range(len(list(n_range))))
    ax1.set_yticks(range(len(list(m_range))))
    ax1.set_xticklabels([str(n) for n in n_range])
    ax1.set_yticklabels([str(m) for m in m_range])
    ax1.set_xlabel("B's Armies (n)", fontsize=11)
    ax1.set_ylabel("A's Armies (m)", fontsize=11)
    ax1.set_title('P(A Wins)', fontsize=13, fontweight='bold')

    for i in range(prob_grid.shape[0]):
        for j in range(prob_grid.shape[1]):
            val = prob_grid[i][j]
            color = DARK['bg'] if 0.35 < val < 0.75 else DARK['text']
            ax1.text(j, i, f'{val:.0%}', ha='center', va='center',
                     fontsize=9, color=color, fontweight='bold')

    ax1.contour(prob_grid, levels=[0.5], colors=['#ffffff'],
                linewidths=2, linestyles='dashed')
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.85)
    cbar1.set_label('P(A Wins)', color=DARK['text'])
    plt.setp(plt.getp(cbar1.ax.axes, 'yticklabels'), color=DARK['dim'])

    cmap_heat = mcolors.LinearSegmentedColormap.from_list(
        'steps', ['#0d2847', '#1158a8', '#58a6ff', DARK['orange'], DARK['red']]
    )
    im2 = ax2.imshow(steps_grid, cmap=cmap_heat, aspect='auto')
    ax2.set_xticks(range(len(list(n_range))))
    ax2.set_yticks(range(len(list(m_range))))
    ax2.set_xticklabels([str(n) for n in n_range])
    ax2.set_yticklabels([str(m) for m in m_range])
    ax2.set_xlabel("B's Armies (n)", fontsize=11)
    ax2.set_ylabel("A's Armies (m)", fontsize=11)
    ax2.set_title('Expected Game Length', fontsize=13, fontweight='bold')

    for i in range(steps_grid.shape[0]):
        for j in range(steps_grid.shape[1]):
            ax2.text(j, i, f'{steps_grid[i][j]:.1f}', ha='center',
                     va='center', fontsize=9, color=DARK['text'], fontweight='bold')

    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.85)
    cbar2.set_label('Expected Steps', color=DARK['text'])
    plt.setp(plt.getp(cbar2.ax.axes, 'yticklabels'), color=DARK['dim'])

    plt.tight_layout()
    return fig


def plot_win_prob_lines(max_m=8, max_n=6):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [DARK['accent'], DARK['green'], DARK['orange'],
              DARK['purple'], DARK['cyan'], DARK['red']]

    for idx, n in enumerate(range(1, max_n + 1)):
        m_vals, p_vals = [], []
        for m in range(2, max_m + 1):
            res = analyze_markov_chain(m, n)
            m_vals.append(m)
            p_vals.append(res['p_a_wins'] * 100)
        ax.plot(m_vals, p_vals, 'o-', color=colors[idx % len(colors)],
                linewidth=2.5, markersize=7, markeredgecolor=DARK['bg'],
                markeredgewidth=1.5, label=f'B = {n}', zorder=3)

    ax.axhline(y=50, color=DARK['dim'], linestyle='--', linewidth=1.5,
               alpha=0.7, label='50% threshold')
    ax.axhspan(50, 100, alpha=0.04, color=DARK['green'])
    ax.axhspan(0, 50, alpha=0.04, color=DARK['red'])
    ax.text(max_m - 0.2, 85, 'A Favored', fontsize=11, color=DARK['green'],
            ha='right', fontstyle='italic', alpha=0.7)
    ax.text(max_m - 0.2, 15, 'B Favored', fontsize=11, color=DARK['red'],
            ha='right', fontstyle='italic', alpha=0.7)

    ax.set_xlabel("A's Armies (m)", fontsize=12)
    ax.set_ylabel("P(A Wins) %", fontsize=12)
    ax.set_title("Win Probability vs A's Army Size", fontsize=13, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9, loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_dice_breakdown():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    scenarios = [
        ("B has 1 army (1d6 vs 1d4)", 1),
        ("B has 2+ armies (1d6 vs 2d4)", 2),
    ]
    for ax, (label, b_armies) in zip(axes, scenarios):
        p_a, p_b, p_t = single_roll_probability(3, b_armies)
        values = [p_a, p_t, p_b]
        names = ['A Wins\nRoll', 'Tie', 'B Wins\nRoll']
        colors = [DARK['green'], DARK['orange'], DARK['red']]
        bars = ax.bar(names, values, color=colors, width=0.6,
                      edgecolor=DARK['border'], linewidth=1.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f'{val:.1%}',
                    ha='center', fontsize=12, fontweight='bold',
                    color=DARK['text'])
        ax.set_ylim(0, 0.65)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
    fig.suptitle('Single Roll Probability Breakdown', fontsize=13,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_expected_steps_bar(max_m=8, max_n=5):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    colors = [DARK['accent'], DARK['green'], DARK['orange'],
              DARK['purple'], DARK['cyan']]
    m_range = list(range(2, max_m + 1))
    n_range = list(range(1, max_n + 1))
    bar_width = 0.15
    x = np.arange(len(m_range))

    for idx, n in enumerate(n_range):
        steps = [analyze_markov_chain(m, n)['expected_steps'] for m in m_range]
        offset = (idx - len(n_range) / 2 + 0.5) * bar_width
        ax.bar(x + offset, steps, bar_width, color=colors[idx % len(colors)],
               label=f'B = {n}', edgecolor=DARK['bg'], linewidth=0.5, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f'A={m}' for m in m_range], fontsize=10)
    ax.set_xlabel("A's Armies", fontsize=12)
    ax.set_ylabel("Expected Steps to Absorption", fontsize=12)
    ax.set_title("Expected Game Length by Configuration", fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


def plot_fundamental_matrix(results):
    N = results['N']
    states = results['states']
    transient = results['transient']
    t_labels = [f"({states[i][0]},{states[i][1]})" for i in transient]

    fig, ax = plt.subplots(figsize=(11, 9))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'purple', ['#0d1117', '#1a1e3a', '#2d1f6e', '#6e40aa', DARK['purple']]
    )
    im = ax.imshow(N, cmap=cmap, aspect='equal')

    ax.set_xticks(range(len(t_labels)))
    ax.set_yticks(range(len(t_labels)))
    ax.set_xticklabels(t_labels, rotation=45, ha='right', fontsize=6)
    ax.set_yticklabels(t_labels, fontsize=6)

    for i in range(N.shape[0]):
        for j in range(N.shape[1]):
            if N[i][j] > 0.05:
                color = DARK['text'] if N[i][j] < np.max(N) * 0.6 else DARK['bg']
                ax.text(j, i, f'{N[i][j]:.2f}', ha='center', va='center',
                        fontsize=5, color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Expected Visits', color=DARK['text'])
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=DARK['dim'])

    ax.set_xlabel('To Transient State', fontsize=11)
    ax.set_ylabel('From Transient State', fontsize=11)
    ax.set_title('Fundamental Matrix N = (I − Q)⁻¹\nExpected visits to state j starting from state i',
                 fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
# STREAMLIT APP LAYOUT
# ═══════════════════════════════════════════════════════════════════════

# Header
st.markdown('<p class="main-header">🎲 Modified Risk — Markov Chain Analysis</p>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">Absorbing Markov chain model for a modified version of Risk '
            '· A rolls 1d6 · B rolls 1d4 or 2d4 · Tie → no army lost</p>',
            unsafe_allow_html=True)

# Sidebar controls
st.sidebar.header("⚙️ Game Parameters")
m = st.sidebar.slider("A's starting armies (m)", min_value=2, max_value=10, value=5)
n = st.sidebar.slider("B's starting armies (n)", min_value=1, max_value=9, value=4)

st.sidebar.markdown("---")
st.sidebar.header("📊 Visualization Options")
max_m_viz = st.sidebar.slider("Max A armies for comparison plots", 4, 12, 8)
max_n_viz = st.sidebar.slider("Max B armies for comparison plots", 3, 9, 7)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📐 Game Rules")
st.sidebar.markdown("""
- **A** always rolls a **6-sided die**
- **B** rolls **1d4** (1 army) or **2d4** (2+ armies)
- A needs **≥2 armies** to continue rolling
- A wins roll → **B loses an army**
- B wins roll → **A loses an army**
- Tie (A = max(B)) → **no change**
- Game ends when **B has 0** (A wins) or **A has 1** (B wins)
""")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Built with Python, NumPy & Matplotlib  \n"
    "Deployed with Streamlit"
)

# Run analysis
results = analyze_markov_chain(m, n)

# ── Summary Stats ──────────────────────────────────────────────────
st.markdown("### 📈 Analysis Results")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("P(A Wins)", f"{results['p_a_wins']:.1%}",
              delta="A favored" if results['p_a_wins'] > 0.5 else "B favored",
              delta_color="normal" if results['p_a_wins'] > 0.5 else "inverse")
with col2:
    st.metric("P(B Wins)", f"{results['p_b_wins']:.1%}")
with col3:
    st.metric("Expected Steps", f"{results['expected_steps']:.1f}",
              help="Average number of dice rolls until the game ends")
with col4:
    matrix_size = results['P'].shape[0]
    st.metric("Matrix Size", f"{matrix_size}×{matrix_size}",
              help=f"{len(results['transient'])} transient + {len(results['absorbing'])} absorbing states")

# ── Math Explanation ───────────────────────────────────────────────
with st.expander("📐 Mathematical Background", expanded=False):
    st.markdown("""
    #### Absorbing Markov Chain Analysis

    The game is modeled as an **absorbing Markov chain** where each state
    represents the current army counts **(a, b)** for players A and B.

    **State Space:** For A starting with *m* armies and B with *n* armies,
    the state space has *m × (n+1)* states. Each state (a, b) can
    transition to:
    - **(a, b−1)** — A wins the roll (probability *pₐ*)
    - **(a−1, b)** — B wins the roll (probability *p_b*)
    - **(a, b)** — Tie, stay in place (probability *p_t*)

    **Absorbing States:** States where **b = 0** (A wins) or **a = 1** (B wins).

    **Canonical Form:** The transition matrix P is decomposed into:
    """)

    st.latex(r"P = \begin{pmatrix} Q & R \\ 0 & I \end{pmatrix}")

    st.markdown("""
    Where **Q** is the transient-to-transient submatrix and **R** is the
    transient-to-absorbing submatrix.

    **Fundamental Matrix:**
    """)

    st.latex(r"N = (I - Q)^{-1}")

    st.markdown("""
    Entry **N[i][j]** gives the expected number of times the chain visits
    transient state *j* starting from transient state *i*.

    **Absorption Probabilities:**
    """)

    st.latex(r"B = N \cdot R")

    st.markdown("""
    Entry **B[i][j]** gives the probability of eventually being absorbed
    into absorbing state *j* starting from transient state *i*.

    **Expected Steps to Absorption:**
    """)

    st.latex(r"t = N \cdot \mathbf{1}")

    st.markdown("""
    The sum of each row of N gives the expected total number of steps
    before the chain reaches any absorbing state.
    """)

# ── Dice Breakdown ─────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🎯 Single Roll Probability Breakdown")
st.markdown("How likely is each outcome on a single dice roll?")

fig_dice = plot_dice_breakdown()
st.pyplot(fig_dice)
plt.close()

# ── Transition Matrix ──────────────────────────────────────────────
st.markdown("---")
st.markdown(f"### 🔢 Transition Matrix P ({results['P'].shape[0]}×{results['P'].shape[0]})")
st.markdown("Each cell shows the probability of moving from one state to another. "
            "Green dashed borders mark absorbing states.")

fig_trans = plot_transition_heatmap(results)
st.pyplot(fig_trans)
plt.close()

# ── Win Probability Lines ──────────────────────────────────────────
st.markdown("---")
st.markdown("### 📉 Win Probability vs Army Size")
st.markdown("How does A's chance of winning change as army counts vary?")

col_left, col_right = st.columns(2)

with col_left:
    fig_lines = plot_win_prob_lines(max_m=max_m_viz, max_n=min(max_n_viz, 6))
    st.pyplot(fig_lines)
    plt.close()

with col_right:
    fig_bars = plot_expected_steps_bar(max_m=max_m_viz, max_n=min(max_n_viz, 5))
    st.pyplot(fig_bars)
    plt.close()

# ── Heatmap Surfaces ──────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🌡️ Win Probability & Game Length — All Configurations")
st.markdown("Left: A's win probability (white dashed line = 50%). "
            "Right: Expected number of rolls until game ends.")

fig_surface = plot_win_probability_surface(max_m=max_m_viz, max_n=max_n_viz)
st.pyplot(fig_surface)
plt.close()

# ── Fundamental Matrix ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🧮 Fundamental Matrix N = (I − Q)⁻¹")
st.markdown("Expected number of visits to each transient state before absorption. "
            "Row sums give expected game length from each starting state.")

fig_fund = plot_fundamental_matrix(results)
st.pyplot(fig_fund)
plt.close()

# ── Raw Data Expander ──────────────────────────────────────────────
st.markdown("---")
with st.expander("🔍 View Raw Matrices", expanded=False):
    tab1, tab2, tab3, tab4 = st.tabs(["P (Full)", "Q (Transient)", "R (Absorbing)", "N (Fundamental)"])

    with tab1:
        labels = [f"({a},{b})" for a, b in results['states']]
        st.dataframe(
            data=dict(zip(labels, results['P'].T)),
            height=400,
        )
    with tab2:
        t_labels = [f"({results['states'][i][0]},{results['states'][i][1]})"
                     for i in results['transient']]
        st.dataframe(
            data=dict(zip(t_labels, results['Q'].T)),
            height=400,
        )
    with tab3:
        a_labels = [f"({results['states'][i][0]},{results['states'][i][1]})"
                     for i in results['absorbing']]
        st.dataframe(
            data=dict(zip(a_labels, results['R'].T)),
            height=400,
        )
    with tab4:
        t_labels = [f"({results['states'][i][0]},{results['states'][i][1]})"
                     for i in results['transient']]
        st.dataframe(
            data=dict(zip(t_labels, results['N'].T)),
            height=400,
        )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #8b949e; font-size: 0.85rem;'>"
    "Modified Risk — Absorbing Markov Chain Analysis · "
    "Built with Python, NumPy, Matplotlib & Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
