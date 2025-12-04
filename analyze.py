import os
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
df = pd.read_csv('training_log.csv')

# Display the first few rows and column info to understand the structure
print(df.head())
print(df.info())
def draw_losses(ax):
    ax.plot(df['iteration'], df['avg_total_loss'], label='Total Loss', color='purple')
    ax.plot(df['iteration'], df['avg_policy_loss'], label='Policy Loss', color='blue')
    ax.plot(df['iteration'], df['avg_value_loss'], label='Value Loss', color='orange')
    ax.set_title('Training Losses')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)


def draw_entropy_confidence(ax):
    ax.plot(df['iteration'], df['avg_policy_entropy'], label='Policy Entropy', color='green')
    ax.set_ylabel('Entropy', color='green')
    ax.tick_params(axis='y', labelcolor='green')
    ax2 = ax.twinx()
    ax2.plot(df['iteration'], df['high_confidence_ratio'], label='High Confidence Ratio', color='red', linestyle='--')
    ax2.set_ylabel('High Confidence Ratio', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax.set_title('Policy Entropy & Confidence')
    ax.set_xlabel('Iteration')
    ax.grid(True)


def draw_outcomes(ax):
    ax.stackplot(
        df['iteration'],
        df['red_win_ratio'],
        df['draw_ratio'],
        df['black_win_ratio'],
        labels=['Red Win', 'Draw', 'Black Win'],
        colors=['#ff9999', '#cccccc', '#666666'],
        alpha=0.7,
    )
    ax.set_title('Game Outcome Ratios')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Ratio')
    ax.legend(loc='upper left')
    ax.grid(True)


def draw_avg_steps(ax):
    ax.plot(df['iteration'], df['avg_game_steps'], label='Avg Game Steps', color='brown')
    ax.set_title('Average Game Steps')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Steps')
    ax.grid(True)


def draw_s1(ax):
    ax.plot(df['iteration'], df['scenario1_value'], label='Scenario 1 Value', color='black', linewidth=2)
    ax.plot(df['iteration'], df['scenario1_masked_a38'], label='Prob Move A38', linestyle='--')
    ax.plot(df['iteration'], df['scenario1_masked_a39'], label='Prob Move A39', linestyle='--')
    ax.plot(df['iteration'], df['scenario1_masked_a40'], label='Prob Move A40', linestyle='--')
    ax.set_title('Scenario 1: Value & Move Probabilities')
    ax.set_xlabel('Iteration')
    ax.legend()
    ax.grid(True)


def draw_s2(ax):
    ax.plot(df['iteration'], df['scenario2_value'], label='Scenario 2 Value', color='black', linewidth=2)
    ax.plot(df['iteration'], df['scenario2_masked_a3'], label='Prob Move A3', linestyle='--')
    ax.plot(df['iteration'], df['scenario2_masked_a5'], label='Prob Move A5', linestyle='--')
    ax.set_title('Scenario 2: Value & Move Probabilities')
    ax.set_xlabel('Iteration')
    ax.legend()
    ax.grid(True)


def draw_buffer(ax):
    ax.plot(df['iteration'], df['replay_buffer_size'], label='Buffer Size', color='teal')
    ax.set_title('Game Buffer Size')
    ax.set_xlabel('Iteration')
    ax.grid(True)


def draw_new_samples(ax):
    ax.plot(df['iteration'], df['new_samples_count'], label='New Samples', color='magenta')
    ax.set_title('New Samples Generated per Iteration')
    ax.set_xlabel('Iteration')
    ax.grid(True)


# ========== 组合大图（原行为保留） ==========
fig, axes = plt.subplots(4, 2, figsize=(20, 24))
plt.subplots_adjust(hspace=0.3)

draw_losses(axes[0, 0])
draw_entropy_confidence(axes[0, 1])
draw_outcomes(axes[1, 0])
draw_avg_steps(axes[1, 1])
draw_s1(axes[2, 0])
draw_s2(axes[2, 1])
draw_buffer(axes[3, 0])
draw_new_samples(axes[3, 1])

fig.savefig('alphazero_analysis.png')
plt.close(fig)


# ========== 分别保存到独立文件 ==========
out_dir = 'analysis_plots'
os.makedirs(out_dir, exist_ok=True)

def save_single(draw_fn, filename, figsize=(10, 6)):
    f, ax = plt.subplots(figsize=figsize)
    draw_fn(ax)
    f.tight_layout()
    f.savefig(os.path.join(out_dir, filename))
    plt.close(f)


save_single(draw_losses, '01_losses.png')
save_single(draw_entropy_confidence, '02_entropy_confidence.png')
save_single(draw_outcomes, '03_outcomes.png')
save_single(draw_avg_steps, '04_avg_steps.png')
save_single(draw_s1, '05_scenario1.png')
save_single(draw_s2, '06_scenario2.png')
save_single(draw_buffer, '07_replay_buffer_size.png')
save_single(draw_new_samples, '08_new_samples_per_iteration.png')