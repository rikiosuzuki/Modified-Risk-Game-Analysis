# 🎲 Modified Risk — Markov Chain Analysis

An absorbing Markov chain model for analyzing a modified version of the board game Risk, computing exact win probabilities and expected game lengths using linear algebra.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://modified-risk-game-analysis-8nmagnso8ehnnn7uhaiyd6.streamlit.app/)

![Win Probability Surface](plots/plot2_win_prob_surface.png)

## Overview

This project models a simplified combat system from Risk as an **absorbing Markov chain**:

- **Player A** rolls a 6-sided die
- **Player B** rolls one 4-sided die (1 army) or two 4-sided dice (2+ armies)
- If A's roll > all of B's → B loses an army
- If A's roll < at least one of B's → A loses an army
- If A's roll = max(B's rolls) → tie, no change
- Game ends when B has 0 armies (A wins) or A has 1 army (B wins)

The **tie mechanic** means the game can theoretically go on forever, making this a proper absorbing Markov chain rather than a simple finite-step process.

## Mathematical Framework

### State Space

Each state is a pair **(a, b)** representing current army counts. For A starting with *m* armies and B with *n*, the transition matrix **P** is *m(n+1) × m(n+1)*.

### Canonical Form Decomposition

The transition matrix is decomposed into:

$$P = \begin{pmatrix} Q & R \\ 0 & I \end{pmatrix}$$

Where:
- **Q** — transient-to-transient transitions (game continues)
- **R** — transient-to-absorbing transitions (game ends)

### Key Computations

| Quantity | Formula | Meaning |
|----------|---------|---------|
| Fundamental Matrix | **N = (I − Q)⁻¹** | Expected visits to each transient state |
| Absorption Probabilities | **B = NR** | Probability of reaching each absorbing state |
| Expected Game Length | **t = N·1** | Expected steps before absorption |

## Visualizations

The project generates 6 publication-quality plots:

| Plot | Description |
|------|-------------|
| **Transition Matrix Heatmap** | Full P matrix with annotated probabilities and absorbing state markers |
| **Win Probability Surface** | Heatmap of P(A wins) across all (m, n) configurations with 50% contour |
| **Expected Game Length Surface** | How many rolls until the game ends for each configuration |
| **Win Probability Lines** | P(A wins) vs A's army count, one line per B army count |
| **Dice Probability Breakdown** | Per-roll outcome probabilities for both B scenarios |
| **Fundamental Matrix N** | Expected state visits heatmap showing chain dynamics |

## Key Results (A=5, B=4)

```
Transition matrix size:  25×25
Transient states:        16
Absorbing states:        9
P(A wins):               71.4%
P(B wins):               28.6%
Expected steps:          6.74 rolls
```

## Project Structure

```
risk-markov-analysis/
├── app.py                  # Streamlit interactive dashboard
├── risk_analysis.py        # Core engine (standalone, no Streamlit)
├── requirements.txt        # Python dependencies
├── plots/                  # Generated visualization PNGs
│   ├── plot1_transition_heatmap.png
│   ├── plot2_win_prob_surface.png
│   ├── plot3_win_prob_lines.png
│   ├── plot4_dice_breakdown.png
│   ├── plot5_expected_steps.png
│   └── plot6_fundamental_matrix.png
└── README.md
```

## Quick Start

### Run the interactive dashboard locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/risk-markov-analysis.git
cd risk-markov-analysis

# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit app
streamlit run app.py
```

### Generate static plots only

```bash
python risk_analysis.py
```

## Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repo → `app.py` as the main file
5. Click **Deploy** — that's it!

Your app will be live at `https://your-app-name.streamlit.app`

## Tech Stack

- **Python** — Core language
- **NumPy** — Matrix operations, linear algebra (matrix inversion)
- **Matplotlib** — Publication-quality visualizations
- **Streamlit** — Interactive web dashboard

## Skills Demonstrated

- **Stochastic Processes** — Markov chain modeling with absorbing states
- **Linear Algebra** — Matrix inversion, fundamental matrix computation
- **Probability Theory** — Combinatorial probability, conditional expectations
- **Scientific Computing** — NumPy vectorized operations
- **Data Visualization** — Custom colormaps, annotated heatmaps, multi-panel figures
- **Web Deployment** — Interactive dashboard with Streamlit

## License

MIT
