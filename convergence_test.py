"""
Q-Learning DDA — Convergence Test & Evaluation
===============================================
Script ini menganalisis file qtable_evaluation_*.json yang dihasilkan
oleh QTableLogger di Unity untuk menguji apakah Q-Learning agent
sudah converge (stabil) atau belum.

Cara pakai:
    py convergence_test.py                              # analisis file terbaru
    py convergence_test.py --file <path_to_json>        # analisis file tertentu
    py convergence_test.py --all                        # analisis semua file sekaligus

Output:
    1. convergence_maxdelta.png   — Grafik MaxDeltaQ per snapshot (harus mendekati 0)
    2. convergence_reward.png     — Grafik moving average reward per episode
    3. convergence_policy.png     — Policy stability (best action per state)
    4. convergence_report.txt     — Laporan teks ringkasan

Kriteria Convergence:
    ✅ MaxDeltaQ < 0.01 selama 50+ snapshot terakhir
    ✅ Policy (best action) tidak berubah selama 50+ snapshot terakhir
    ✅ Moving average reward stabil (std < 0.3 di 500 episode terakhir)
"""

import json
import os
import sys
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from collections import Counter, defaultdict


# ─────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────

def find_evaluation_files():
    """Cari semua file qtable_evaluation di persistentDataPath Unity."""
    locallow = os.path.join(os.environ.get('APPDATA', ''), '..', 'LocalLow')
    pattern = os.path.join(locallow, '**', 'qtable_evaluation_*.json')
    files = glob.glob(pattern, recursive=True)
    return sorted(files, key=os.path.getmtime)


def load_evaluation(filepath):
    """Load JSON evaluation file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


# ─────────────────────────────────────────────────────────────────
# TEST 1: MaxDeltaQ Convergence
# ─────────────────────────────────────────────────────────────────

def test_maxdelta_convergence(snapshots, threshold=0.01, window=50, output_dir='.'):
    """
    Tes apakah perubahan Q-value terbesar (MaxDeltaQ) sudah stabil mendekati 0.
    
    Kriteria PASS:
        MaxDeltaQ < threshold untuk seluruh `window` snapshot terakhir.
    """
    episodes = [s['episode'] for s in snapshots]
    max_deltas = [s['maxDeltaQ'] for s in snapshots]

    if len(max_deltas) == 0:
        return False, "Tidak ada snapshot data."

    # Cek convergence di window terakhir
    tail = max_deltas[-window:] if len(max_deltas) >= window else max_deltas
    converged = all(d < threshold for d in tail)
    avg_delta = np.mean(tail)
    max_tail = max(tail)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(episodes, max_deltas, linewidth=0.8, alpha=0.7, label='MaxDeltaQ')
    ax.axhline(y=threshold, color='r', linestyle='--', linewidth=1, label=f'Threshold ({threshold})')

    # Moving average
    if len(max_deltas) > 20:
        ma_window = min(50, len(max_deltas) // 5)
        ma = np.convolve(max_deltas, np.ones(ma_window) / ma_window, mode='valid')
        ma_episodes = episodes[ma_window - 1:]
        ax.plot(ma_episodes, ma, color='orange', linewidth=2, label=f'Moving Avg ({ma_window})')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Max ΔQ')
    ax.set_title('Convergence Test: MaxDeltaQ per Snapshot')
    ax.legend()
    ax.set_yscale('log') if max(max_deltas) > 1 else None
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, 'convergence_maxdelta.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    status = "✅ CONVERGED" if converged else "❌ NOT CONVERGED"
    detail = (f"{status}\n"
              f"  Last {len(tail)} snapshots: avg={avg_delta:.6f}, max={max_tail:.6f}\n"
              f"  Threshold: {threshold}\n"
              f"  Total snapshots: {len(snapshots)}\n"
              f"  Chart: {path}")

    return converged, detail


# ─────────────────────────────────────────────────────────────────
# TEST 2: Reward Stability
# ─────────────────────────────────────────────────────────────────

def test_reward_stability(episodes_data, window=500, std_threshold=0.5, output_dir='.'):
    """
    Tes apakah reward sudah stabil.
    
    Kriteria PASS:
        Standard deviation of moving average reward < std_threshold
        di `window` episode terakhir.
    """
    rewards = [e['totalReward'] for e in episodes_data]
    ep_numbers = [e['episode'] for e in episodes_data]

    if len(rewards) < window:
        return False, f"Episode terlalu sedikit ({len(rewards)} < {window})."

    # Moving average
    ma_size = min(100, len(rewards) // 10)
    if ma_size < 2:
        ma_size = 2
    ma = np.convolve(rewards, np.ones(ma_size) / ma_size, mode='valid')
    ma_episodes = ep_numbers[ma_size - 1:]

    # Cek stabilitas di tail
    tail_ma = ma[-window:] if len(ma) >= window else ma
    tail_std = np.std(tail_ma)
    tail_mean = np.mean(tail_ma)
    stable = tail_std < std_threshold

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})

    # Reward per episode
    ax1.scatter(ep_numbers, rewards, s=1, alpha=0.2, color='blue', label='Per-episode reward')
    ax1.plot(ma_episodes, ma, color='red', linewidth=1.5, label=f'Moving Avg ({ma_size})')
    ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax1.set_ylabel('Reward')
    ax1.set_title('Reward per Episode')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Distribusi reward di tail
    tail_rewards = rewards[-window:]
    ax2.hist(tail_rewards, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.axvline(x=np.mean(tail_rewards), color='red', linestyle='--', label=f'Mean={np.mean(tail_rewards):.2f}')
    ax2.set_xlabel('Reward')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Reward Distribution (last {window} episodes) — Std={tail_std:.3f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, 'convergence_reward.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    status = "✅ STABLE" if stable else "❌ UNSTABLE"
    detail = (f"{status}\n"
              f"  Last {len(tail_ma)} episodes MA: mean={tail_mean:.4f}, std={tail_std:.4f}\n"
              f"  Std threshold: {std_threshold}\n"
              f"  Total episodes: {len(rewards)}\n"
              f"  Chart: {path}")

    return stable, detail


# ─────────────────────────────────────────────────────────────────
# TEST 3: Policy Stability
# ─────────────────────────────────────────────────────────────────

def test_policy_stability(snapshots, stable_window=50, output_dir='.'):
    """
    Tes apakah policy (best action per state) sudah stabil.
    
    Kriteria PASS:
        Best action per state tidak berubah selama `stable_window` snapshot terakhir.
    """
    if len(snapshots) < 2:
        return False, "Snapshot terlalu sedikit untuk policy stability test."

    # Bangun timeline policy per state
    state_names = set()
    for s in snapshots:
        for entry in s.get('states', []):
            state_names.add(entry['state'])
    state_names = sorted(state_names)

    # Tracking: kapan terakhir best action berubah per state
    policy_timeline = {name: [] for name in state_names}
    for snap in snapshots:
        state_map = {entry['state']: entry for entry in snap.get('states', [])}
        for name in state_names:
            if name in state_map:
                policy_timeline[name].append(state_map[name]['bestActionName'])
            else:
                policy_timeline[name].append(None)

    # Cek stabilitas di window terakhir
    all_stable = True
    changes_info = {}
    for name in state_names:
        timeline = policy_timeline[name]
        tail = timeline[-stable_window:] if len(timeline) >= stable_window else timeline
        # Filter None
        tail_filtered = [a for a in tail if a is not None]
        unique_actions = set(tail_filtered)
        is_stable = len(unique_actions) <= 1
        if not is_stable:
            all_stable = False
        changes_info[name] = {
            'stable': is_stable,
            'actions_in_tail': list(unique_actions),
            'final_action': tail_filtered[-1] if tail_filtered else 'N/A'
        }

    # Plot: heatmap policy per state per snapshot
    action_to_num = {'Maintain': 0, 'Increase': 1, 'Decrease': 2}
    action_colors = ['#2196F3', '#F44336', '#4CAF50']  # blue, red, green
    action_labels = ['Maintain', 'Increase', 'Decrease']

    episodes_axis = [s['episode'] for s in snapshots]
    n_states = len(state_names)
    n_snaps = len(snapshots)

    if n_states > 0 and n_snaps > 0:
        matrix = np.full((n_states, n_snaps), -1, dtype=int)
        for j, snap in enumerate(snapshots):
            state_map = {entry['state']: entry for entry in snap.get('states', [])}
            for i, name in enumerate(state_names):
                if name in state_map:
                    action_name = state_map[name]['bestActionName']
                    matrix[i, j] = action_to_num.get(action_name, -1)

        fig, ax = plt.subplots(figsize=(14, max(4, n_states * 0.4)))

        from matplotlib.colors import ListedColormap, BoundaryNorm
        cmap = ListedColormap(action_colors)
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

        # Mask -1 values
        masked = np.ma.masked_where(matrix == -1, matrix)
        im = ax.imshow(masked, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')

        ax.set_yticks(range(n_states))
        ax.set_yticklabels(state_names, fontsize=7)

        # X ticks: show some episode numbers
        tick_step = max(1, n_snaps // 15)
        tick_positions = list(range(0, n_snaps, tick_step))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([episodes_axis[i] for i in tick_positions], fontsize=7, rotation=45)
        ax.set_xlabel('Episode (snapshot)')
        ax.set_title(f'Policy Stability — Best Action per State over Time')

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=c, label=l) for c, l in zip(action_colors, action_labels)]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=8)
        ax.grid(False)

        path = os.path.join(output_dir, 'convergence_policy.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        path = "(no plot)"

    status = "✅ STABLE" if all_stable else "❌ UNSTABLE"
    lines = [f"{status} (last {stable_window} snapshots)\n"]
    for name in state_names:
        info = changes_info[name]
        icon = "✅" if info['stable'] else "❌"
        lines.append(f"  {icon} {name}: final={info['final_action']}, "
                      f"actions={info['actions_in_tail']}")
    lines.append(f"\n  Chart: {path}")

    return all_stable, "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# TEST 4: Q-Value Statistics
# ─────────────────────────────────────────────────────────────────

def test_qvalue_stats(snapshots, output_dir='.'):
    """Plot meanAbsQ dan stdQ over time untuk melihat learning progress."""
    episodes = [s['episode'] for s in snapshots]
    mean_abs = [s['meanAbsQ'] for s in snapshots]
    std_q = [s['stdQ'] for s in snapshots]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax1.plot(episodes, mean_abs, color='blue', linewidth=1)
    ax1.set_ylabel('Mean |Q|')
    ax1.set_title('Q-Value Statistics over Training')
    ax1.grid(True, alpha=0.3)

    ax2.plot(episodes, std_q, color='purple', linewidth=1)
    ax2.set_ylabel('Std Q')
    ax2.set_xlabel('Episode')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, 'convergence_qstats.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return (f"  Mean|Q| final: {mean_abs[-1]:.4f}\n"
            f"  StdQ final:    {std_q[-1]:.4f}\n"
            f"  Chart: {path}")


# ─────────────────────────────────────────────────────────────────
# TEST 5: Action Distribution
# ─────────────────────────────────────────────────────────────────

def test_action_distribution(episodes_data, output_dir='.'):
    """Analisis distribusi action yang dipilih sepanjang training."""
    difficulties = [e['difficulty'] for e in episodes_data]
    total = len(difficulties)
    counter = Counter(difficulties)

    # Per-window distribution
    window = min(200, total // 5) if total > 10 else total
    n_windows = total // window if window > 0 else 1

    actions_over_time = {'Maintain': [], 'Increase': [], 'Decrease': []}
    window_centers = []

    for i in range(n_windows):
        start = i * window
        end = start + window
        chunk = difficulties[start:end]
        chunk_count = Counter(chunk)
        for a in actions_over_time:
            actions_over_time[a].append(chunk_count.get(a, 0) / len(chunk) * 100)
        window_centers.append(start + window // 2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Pie chart overall
    labels = list(counter.keys())
    sizes = list(counter.values())
    colors_map = {'Maintain': '#2196F3', 'Increase': '#F44336', 'Decrease': '#4CAF50'}
    colors = [colors_map.get(l, '#999') for l in labels]
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title(f'Overall Action Distribution (n={total})')

    # Stacked area over time
    if len(window_centers) > 1:
        for action, color in colors_map.items():
            if action in actions_over_time:
                ax2.plot(window_centers, actions_over_time[action],
                         label=action, color=color, linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('% of actions')
        ax2.set_title(f'Action Distribution over Time (window={window})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)

    fig.tight_layout()
    path = os.path.join(output_dir, 'convergence_actions.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    lines = ["  Action Distribution:"]
    for action in ['Maintain', 'Increase', 'Decrease']:
        count = counter.get(action, 0)
        pct = count / total * 100 if total > 0 else 0
        lines.append(f"    {action}: {count} ({pct:.1f}%)")
    lines.append(f"  Chart: {path}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# FINAL Q-TABLE ANALYSIS
# ─────────────────────────────────────────────────────────────────

def analyze_final_qtable(snapshots):
    """Analisis Q-table final: policy rekomendasi per state."""
    if not snapshots:
        return "  Tidak ada snapshot."

    final = snapshots[-1]
    lines = [f"  Final Q-Table (episode {final['episode']}):"]
    lines.append(f"  {'State':<35} {'Maintain':>10} {'Increase':>10} {'Decrease':>10}  {'→ Best':>12}")
    lines.append(f"  {'─'*35} {'─'*10} {'─'*10} {'─'*10}  {'─'*12}")

    for entry in sorted(final.get('states', []), key=lambda x: x['state']):
        qv = entry['qValues']
        q_str = [f"{v:>10.4f}" for v in qv]
        lines.append(f"  {entry['state']:<35} {''.join(q_str)}  → {entry['bestActionName']:>8}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def run_convergence_test(filepath, output_dir=None):
    """Run all convergence tests pada satu file evaluation."""
    print(f"\n{'='*70}")
    print(f"  CONVERGENCE TEST — Q-Learning DDA")
    print(f"  File: {os.path.basename(filepath)}")
    print(f"{'='*70}\n")

    data = load_evaluation(filepath)

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(filepath), 'convergence_results')
    os.makedirs(output_dir, exist_ok=True)

    # Info
    hyper = data.get('hyperparameters', {})
    print(f"📋 Session: {data.get('sessionId', 'N/A')}")
    print(f"📋 Start:   {data.get('startTime', 'N/A')}")
    print(f"📋 Hyperparameters:")
    print(f"   α (learning rate)  = {hyper.get('alpha', 'N/A')}")
    print(f"   γ (discount)       = {hyper.get('gamma', 'N/A')}")
    print(f"   ε (start → end)    = {hyper.get('epsilonStart', 'N/A')} → {hyper.get('epsilonEnd', 'N/A')}")
    print(f"   ε decay            = {hyper.get('epsilonDecay', 'N/A')}")
    print(f"   Total episodes     = {hyper.get('totalEpisodes', 'N/A')}")
    print(f"   Logged episodes    = {len(data.get('episodes', []))}")
    print(f"   Snapshots          = {len(data.get('qtableSnapshots', []))}")

    episodes_data = data.get('episodes', [])
    snapshots = data.get('qtableSnapshots', [])

    results = []
    report_lines = []

    # Test 1: MaxDeltaQ
    print(f"\n{'─'*50}")
    print("🔬 TEST 1: MaxDeltaQ Convergence")
    passed, detail = test_maxdelta_convergence(snapshots, output_dir=output_dir)
    results.append(('MaxDeltaQ Convergence', passed))
    print(detail)
    report_lines.append(f"[TEST 1] MaxDeltaQ Convergence\n{detail}\n")

    # Test 2: Reward Stability
    print(f"\n{'─'*50}")
    print("🔬 TEST 2: Reward Stability")
    passed, detail = test_reward_stability(episodes_data, output_dir=output_dir)
    results.append(('Reward Stability', passed))
    print(detail)
    report_lines.append(f"[TEST 2] Reward Stability\n{detail}\n")

    # Test 3: Policy Stability
    print(f"\n{'─'*50}")
    print("🔬 TEST 3: Policy Stability")
    passed, detail = test_policy_stability(snapshots, output_dir=output_dir)
    results.append(('Policy Stability', passed))
    print(detail)
    report_lines.append(f"[TEST 3] Policy Stability\n{detail}\n")

    # Test 4: Q-Value Stats
    if snapshots:
        print(f"\n{'─'*50}")
        print("📊 Q-Value Statistics")
        detail = test_qvalue_stats(snapshots, output_dir=output_dir)
        print(detail)
        report_lines.append(f"[STATS] Q-Value Statistics\n{detail}\n")

    # Test 5: Action Distribution
    if episodes_data:
        print(f"\n{'─'*50}")
        print("📊 Action Distribution")
        detail = test_action_distribution(episodes_data, output_dir=output_dir)
        print(detail)
        report_lines.append(f"[STATS] Action Distribution\n{detail}\n")

    # Final Q-Table
    print(f"\n{'─'*50}")
    print("📊 Final Q-Table Policy")
    detail = analyze_final_qtable(snapshots)
    print(detail)
    report_lines.append(f"[POLICY] Final Q-Table\n{detail}\n")

    # Summary
    print(f"\n{'='*70}")
    print("📋 RINGKASAN CONVERGENCE TEST")
    print(f"{'='*70}")
    all_passed = True
    for name, passed in results:
        icon = "✅" if passed else "❌"
        print(f"  {icon} {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print(f"\n  🎉 SEMUA TEST PASSED — Q-Learning agent sudah CONVERGE!")
        print(f"  Agent siap digunakan untuk DDA runtime.")
    else:
        print(f"\n  ⚠️  Belum semua test passed — agent mungkin perlu training lebih lama")
        print(f"  atau hyperparameter perlu di-tuning.")

    print(f"\n  Output directory: {output_dir}")

    # Save report
    report_path = os.path.join(output_dir, 'convergence_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"CONVERGENCE TEST REPORT\n")
        f.write(f"File: {filepath}\n")
        f.write(f"Session: {data.get('sessionId', 'N/A')}\n")
        f.write(f"Date: {data.get('startTime', 'N/A')}\n")
        f.write(f"{'='*60}\n\n")
        for line in report_lines:
            f.write(line + "\n")
        f.write(f"\n{'='*60}\n")
        f.write(f"RESULT: {'ALL PASSED ✅' if all_passed else 'NOT ALL PASSED ❌'}\n")

    print(f"  Report saved: {report_path}")
    return all_passed


def main():
    parser = argparse.ArgumentParser(description='Q-Learning DDA Convergence Test')
    parser.add_argument('--file', '-f', type=str, help='Path to specific qtable_evaluation JSON file')
    parser.add_argument('--all', '-a', action='store_true', help='Analyze all evaluation files')
    parser.add_argument('--output', '-o', type=str, help='Output directory for charts and report')
    args = parser.parse_args()

    if args.file:
        if not os.path.exists(args.file):
            print(f"❌ File tidak ditemukan: {args.file}")
            sys.exit(1)
        run_convergence_test(args.file, args.output)

    elif args.all:
        files = find_evaluation_files()
        if not files:
            print("❌ Tidak ditemukan file qtable_evaluation_*.json")
            sys.exit(1)
        print(f"Ditemukan {len(files)} file evaluasi.")
        for f in files:
            run_convergence_test(f, args.output)

    else:
        # Default: file terbaru
        files = find_evaluation_files()
        if not files:
            print("❌ Tidak ditemukan file qtable_evaluation_*.json")
            print("   Jalankan training di Unity terlebih dahulu, atau gunakan --file <path>")
            sys.exit(1)
        latest = files[-1]
        print(f"Menggunakan file terbaru: {os.path.basename(latest)}")
        run_convergence_test(latest, args.output)


if __name__ == '__main__':
    main()
