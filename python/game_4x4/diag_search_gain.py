"""诊断：模型的搜索增益（统一评估协议，见 eval_common）。

用法：python diag_search_gain.py [model.pt] [n]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

torch.set_num_threads(int(os.getenv("G4X4_TORCH_THREADS", "2")))

from eval_common import (
    load_predictor, play_match, model_mcts_action, heuristic_action,
    EVAL_SIMS,
)


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    pred = load_predictor(model_path)

    def obs_of(env):
        import numpy as np
        b, s = env.observation()
        return (np.asarray(b, dtype=np.float32).reshape(1, 16, 4, 4),
                np.asarray(s, dtype=np.float32).reshape(1, -1))

    def greedy(env):
        logits, _ = pred(*obs_of(env))
        return max(env.legal_moves(), key=lambda a: logits[0][a])

    def mcts(env, sims):
        return model_mcts_action(env, pred, sims)

    def hm(env):
        return heuristic_action(env)

    print("=== 搜索增益 vs 启发式64（交替先后手）===", flush=True)

    def report_fn(fn, tag):
        wins = draws = losses = 0
        model_is_red = True
        for _ in range(n):
            env = None
            import banqi_4x8
            env = banqi_4x8.Game4x4()
            moves = 0
            while not env.terminated():
                if env.winner() is not None:
                    break
                cur = env.current_player()
                if (cur == 1) == model_is_red:
                    a = fn(env)
                else:
                    a = hm(env)
                if a is None:
                    break
                env.step(a)
                moves += 1
                if moves > 400:
                    break
            w = env.winner()
            if w == 0:
                draws += 1
            elif (w == 1) == model_is_red:
                wins += 1
            else:
                losses += 1
            model_is_red = not model_is_red
        print(f"  {tag}: 胜{wins} 平{draws} 负{losses} = {100*wins/n:.0f}%", flush=True)

    report_fn(lambda e: greedy(e), "greedy(先验)")
    report_fn(lambda e: mcts(e, 64), "MCTS64(搜索)")
    report_fn(lambda e: mcts(e, 256), "MCTS256(搜索)")


if __name__ == "__main__":
    main()
