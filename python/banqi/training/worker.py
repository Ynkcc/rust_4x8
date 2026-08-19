"""banqi/training/worker.py — 训练 worker 线程。

TrainWorker 在独立线程消费 self_play 队列，把 episode 转换的 sample 写入
DataBuffer，按 (new_samples/batch)×epochs 限制训练量（避免旧数据反复训练），
并在 checkpoint 时保存 model/optimizer/scheduler/global_step + 训练监控。
"""

from __future__ import annotations

import os
import random
import time
import copy
import pickle
import threading
from collections import deque
from typing import Dict, List, Optional

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from banqi.checkpoint import export_torchscript, export_onnx, export_model_isolated
from banqi.constants import build_constants
from banqi.tb_logger import add_scalar
from banqi.variant import Variant

try:
    import banqi_4x8
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "无法导入 banqi_4x8。请先执行: maturin develop --features pyo3"
    ) from exc

from .buffer import DataBuffer, episode_to_samples
from .losses import run_training_epochs, _resolve_device
from .eval import (
    build_fixed_eval,
    eval_match,
    eval_policy_accuracy,
    eval_value_drift,
    prefill_from_archive,
    select_balanced_fixed_samples,
)


def _compute_lr_scale(global_step: int, cfg) -> float:
    """线性 warmup 系数：前 LR_DECAY_STEPS 步从 MIN_LR 线性升到 1.0。

    仅用于 checkpoint 恢复时对初始 lr 做缩放，避免从 0 直接起训导致的震荡；
    之后的余弦下降交由 CosineAnnealingLR 处理。
    """
    if global_step <= 0:
        return 1.0
    decay_steps = max(int(getattr(cfg, "LR_DECAY_STEPS", 1000) or 1000), 1)
    min_lr = float(getattr(cfg, "MIN_LR", 1e-6) or 1e-6)
    frac = min(global_step, decay_steps) / decay_steps
    # 目标 = MIN_LR + (1 - MIN_LR)*frac，最小不低于 MIN_LR
    return max(min_lr, 1.0 - (1.0 - min_lr) * (1.0 - frac))


def _is_stopped(stop_event) -> bool:
    if stop_event is None:
        return False
    if isinstance(stop_event, list):
        return bool(stop_event[0])
    if hasattr(stop_event, "is_set"):
        return stop_event.is_set()
    return bool(stop_event)


def _make_cosine_clamp_scheduler(optimizer, cfg):
    """余弦衰减到 MIN_LR 后钳位保持，不周期回升。

    原生 CosineAnnealingLR 在训练步数超过 T_max 后学习率会按余弦周期回升，
    导致长周期自对弈训练后期梯度偏大、收敛震荡。这里用 LambdaLR 实现：
    前 LR_DECAY_STEPS 步按半周期余弦从 LEARNING_RATE 平滑降到 MIN_LR，
    之后钳位在 MIN_LR 保持，兼顾余弦退火的平滑收敛与长训练稳定性。
    """
    t_max = max(int(getattr(cfg, "LR_DECAY_STEPS", 1000) or 1000), 1)
    eta_min = float(getattr(cfg, "MIN_LR", 1e-6) or 1e-6)
    eta_max = float(getattr(cfg, "LEARNING_RATE", 1e-4) or 1e-4)
    # LambdaLR 的 lambda 返回的是相对 initial_lr 的比例因子
    min_ratio = eta_min / eta_max if eta_max > 0 else 1e-4

    def lr_lambda(epoch: int) -> float:
        import math
        t = min(epoch, t_max) / t_max            # 钳位到 [0,1]
        # 半周期余弦：t=0 -> 1.0，t=1 -> 0.0（即 MIN_LR）
        cos = 0.5 * (1.0 + math.cos(math.pi * t))
        return min_ratio + (1.0 - min_ratio) * cos

    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


class TrainWorker(threading.Thread):
    def __init__(self, arg1, arg2, arg3=None, arg4=None,
                 ckpt_dir: Optional[str] = None, device=None, run_dir: Optional[str] = None):
        if isinstance(arg1, Variant):
            variant = arg1
            cfg = arg2
            data_queue = arg3
            stop_event = arg4
        elif isinstance(arg3, Variant):
            # 兼容 (data_q, stop_flag, variant) 参数顺序
            data_queue = arg1
            stop_event = arg2
            variant = arg3
            from banqi.config import make_config
            cfg = arg4 if arg4 is not None else make_config(variant.id)
        else:
            # 兼容 (data_q, stop_flag, variant_id_str)
            data_queue = arg1
            stop_event = arg2
            from banqi.variant import get_variant
            from banqi.config import make_config
            variant = get_variant(str(arg3))
            cfg = arg4 if arg4 is not None else make_config(variant.id)

        super().__init__(name=f"TrainWorker-{variant.id}", daemon=True)
        self.variant = variant
        self.cfg = cfg
        self.C = build_constants(variant)
        self.data_queue = data_queue
        self.stop_event = stop_event
        self.ckpt_dir = ckpt_dir or variant.checkpoints_dir
        self.run_dir = run_dir
        self.device = device or _resolve_device(getattr(cfg, "TRAIN_DEVICE", "auto"))
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # 监控：每轮训练时长、最近 ckpt 路径、最近一次 epoch loss 分解
        self.metrics = {
            "train_duration": 0.0,
            "last_ckpt_path": None,
            "last_epoch_losses": None,
            "last_lr": 0.0,
            "global_step": 0,
        }

        # S3 默认 CPU（避免与主训练 GPU 争抢）；GPU 推理时显式 device
        self.desired_sp_device = (
            "cuda" if getattr(cfg, "SELF_PLAY_DEVICE", "cpu").startswith("cuda") else "cpu"
        )

        self._last_ckpt_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self.round_num = 0
        self.total_loss_sum = 0.0
        self.total_policy_loss_sum = 0.0
        self.total_value_loss_sum = 0.0
        self.round_history: deque = deque(maxlen=1000)

        self._warmup_done = False
        self._raw_sample_pool: List[Dict] = []
        self._fixed_eval: Optional[Dict] = None
        self._prev_weights: Optional[Dict[str, torch.Tensor]] = None
        # 动作置换表缓存（Rust 绑定导出），增强用
        self._perm_cache: Dict[str, list] = {}
        self._init_model_and_checkpoint()

        # 检查是否启动后台 gRPC 服务端（独立自对弈架构支持）
        # 训练端同时作为 gRPC Server（提供模型/控制命令、接收元信息）与
        # Client（主动拉取 Rust Worker 产生的样本）。不依赖 Rust PyO3 绑定。
        self.grpc_server_thread = None
        grpc_cfg = getattr(self.cfg, "grpc_server", None) or {}
        if isinstance(grpc_cfg, dict) and grpc_cfg.get("ENABLED", False):
            from banqi.grpc_server import GrpcServerThread
            host = grpc_cfg.get("HOST", "0.0.0.0")
            port = int(grpc_cfg.get("PORT", 50051))
            max_workers = int(grpc_cfg.get("MAX_WORKERS", 10))
            worker_host = grpc_cfg.get("WORKER_HOST")
            worker_port = int(grpc_cfg.get("WORKER_PORT", 50052))
            pull_enabled = bool(grpc_cfg.get("PULL_ENABLED", True))

            def get_model_path():
                # 提供 TorchScript 模型（worker 拉取后由 tch CModule 加载）
                pt = os.path.join(self.ckpt_dir, "last.pt")
                return pt if os.path.exists(pt) else None

            def get_config():
                return {
                    "mcts_sims": getattr(self.cfg, "MCTS_SIMS", 128),
                    "temperature": 1.0,
                    "playout_cap_random": getattr(self.cfg, "PLAYOUT_CAP_RANDOM_ENABLED", False),
                }

            self.grpc_server_thread = GrpcServerThread(
                host=host,
                port=port,
                max_workers=max_workers,
                worker_host=worker_host,
                worker_port=worker_port,
                pull_enabled=pull_enabled,
                data_queue=self.data_queue,
                model_path_provider=get_model_path,
                config_provider=get_config,
            )
            self.grpc_server_thread.start()

    def _init_model_and_checkpoint(self):
        cfg = self.cfg
        from banqi.nn_model import BanqiNet

        ema_enabled = getattr(cfg, "EMA_ENABLED", True)
        ema_decay = float(getattr(cfg, "EMA_DECAY", 0.999))
        self.ema_enabled = ema_enabled
        self.ema_decay = ema_decay
        self.ema_model = None

        if os.path.exists(self.last_ckpt_path()):  # resume
            print(f"[TR-{self.variant.id}] 从 checkpoint 恢复: {self.last_ckpt_path()}")
            ckpt = torch.load(self.last_ckpt_path(), map_location=self.device, weights_only=False)
            model = BanqiNet(self.variant)
            model.load_state_dict(ckpt["model_state"])
            self.model = model.to(self.device)
            if ema_enabled:
                self.ema_model = BanqiNet(self.variant).to(self.device)
                if "ema_model_state" in ckpt and ckpt["ema_model_state"] is not None:
                    self.ema_model.load_state_dict(ckpt["ema_model_state"])
                else:
                    self.ema_model.load_state_dict(self.model.state_dict())
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=cfg.LEARNING_RATE * _compute_lr_scale(ckpt.get("global_step", 0), cfg),
                weight_decay=cfg.WEIGHT_DECAY,
            )
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
            self.scheduler = _make_cosine_clamp_scheduler(self.optimizer, cfg)
            if "scheduler_state" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler_state"])
            self.global_step = ckpt.get("global_step", 0)
            self.metrics["global_step"] = self.global_step
            self.start_global_step = self.global_step
            self.start_total_samples = ckpt.get("total_samples", 0)
            self.version = ckpt.get("version", 0) + 1
            print(f"[TR-{self.variant.id}] 恢复 global_step={self.global_step}, "
                  f"version={self.version}" + (" (EMA 已启用)" if ema_enabled else ""))
        else:
            self.model = BanqiNet(self.variant).to(self.device)
            if ema_enabled:
                self.ema_model = BanqiNet(self.variant).to(self.device)
                self.ema_model.load_state_dict(self.model.state_dict())
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=cfg.LEARNING_RATE,
                weight_decay=cfg.WEIGHT_DECAY
            )
            self.scheduler = _make_cosine_clamp_scheduler(self.optimizer, cfg)
            self.global_step = 0
            self.start_global_step = 0
            self.start_total_samples = 0
            self.version = 0

            # 冷启动：立即导出初始模型，供 Rust 自对弈加载，避免自对弈等待
            # last.pt、训练 worker 又等待自对弈数据，两者互相等待而死锁。
            self._export_initial_model()

        buffer_capacity = cfg.MAX_SAMPLE_BUFFER_SIZE
        self.buffer = DataBuffer(buffer_capacity, self.variant, cfg)

        # 冷存储预填充 & 固定验证集生成
        fixed_archive = prefill_from_archive(self.buffer, self.variant, cfg)
        if fixed_archive is not None:
            self._fixed_eval = fixed_archive

        # 显式记录 value 目标模式（终端日志，便于复现/调试）
        print(f"[TR-{self.variant.id}] 价值目标模式={cfg.VALUE_TARGET_MODE}，"
              f"buffer 容量={buffer_capacity}，TRAIN_DEVICE={self.device}")

        # ---- value 目标退火（VALUE_TARGET_MODE='anneal' 时）----
        # 退火权重 w：前 N 轮用 mcts 平滑评估，后段切到 game_result 真值。
        # 每轮训练前按 (round_idx / anneal_rounds) 更新 buffer.value_result_weight。
        self.anneal_rounds = getattr(cfg, "VALUE_TARGET_ANNEAL_ROUNDS", 0)

        init_ckpt = getattr(cfg, "INIT_FROM_CHECKPOINT", None)
        if init_ckpt:
            self._load_pretrained(init_ckpt)

    def _load_pretrained(self, ckpt_path: str):
        """从指定 checkpoint 导入权重（仅 model + optimizer），重置 global_step。"""
        print(f"[TR-{self.variant.id}] 加载预训练权重: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        try:
            self.model.load_state_dict(ckpt["model_state"])
        except RuntimeError as e:
            print(f"[TR-{self.variant.id}] ⚠️ state_dict 不完全匹配（跨变体/结构），"
                  f"忽略不匹配键: {e}")
            sd = ckpt["model_state"]
            own = self.model.state_dict()
            filtered = {k: v for k, v in sd.items()
                        if k in own and v.shape == own[k].shape}
            own.update(filtered)
            self.model.load_state_dict(own)
        if "optimizer_state" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state"])
            except Exception:
                pass
        self.global_step = 0
        self.metrics["global_step"] = 0

    def last_ckpt_path(self):
        return os.path.join(self.ckpt_dir, "last.ckpt")

    def _export_initial_model(self) -> None:
        """冷启动时导出初始模型（.pt/.onnx），供 Rust 自对弈加载。

        全新起训时 last.pt 不存在，若不自对弈先导出，Rust 自对弈会一直等待
        last.pt，而训练 worker 又因拿不到自对弈数据永不导出，形成死锁。
        这里在模型初始化后立即导出一次初始权重，打破该循环依赖。
        """
        pt_path = os.path.join(self.ckpt_dir, "last.pt")
        onnx_path = os.path.join(self.ckpt_dir, "last.onnx")
        if os.path.exists(pt_path):
            return
        try:
            self.model.eval()
            export_model_isolated(self.model, pt_path, onnx_path, self.variant, self.device)
            print(f"[TR-{self.variant.id}] 💾 冷启动：已导出初始模型 {pt_path}")
        except Exception as exc:  # noqa: BLE001 - 初始导出失败不阻塞启动，训练仍可推进
            print(f"[TR-{self.variant.id}] ⚠️ 冷启动初始模型导出失败: {exc}")

    def get_inference_model(self):
        if self.ema_enabled and self.ema_model is not None:
            self.ema_model.eval()
            return self.ema_model
        return self.model

    def get_global_step(self):
        return self.global_step

    def get_model_version(self):
        return self.version

    def get_checkpoint_path(self):
        return self.last_ckpt_path()

    def save_checkpoint(
        self,
        new_samples: int = 0,
        total_samples: Optional[int] = None,
        round_idx: int = 0,
        force: bool = False,
    ) -> None:
        save_every = max(int(getattr(self.cfg, "CKPT_SAVE_EVERY", 1)), 1)
        export_every = max(int(getattr(self.cfg, "CKPT_EXPORT_EVERY", 10)), 1)

        pt_path = os.path.join(self.ckpt_dir, "last.pt")
        onnx_path = os.path.join(self.ckpt_dir, "last.onnx")

        should_save_ckpt = force or (round_idx % save_every == 0) or (round_idx == 0)
        # round_idx==0 时仅在 pt 不存在时导出一次；round_idx>0 才按周期导出，
        # 避免冷启动/rule_selfplay 早期 round 固定时每个训练步都重导 TorchScript/ONNX 拖慢吞吐。
        should_export = force or (round_idx > 0 and round_idx % export_every == 0) \
            or not os.path.exists(pt_path)

        if not should_save_ckpt and not should_export:
            return

        path = self.last_ckpt_path()
        if should_save_ckpt:
            self.model.eval()
            if self.ema_model is not None:
                self.ema_model.eval()
            snapshot = {
                "model_state": self.model.state_dict(),
                "ema_model_state": self.ema_model.state_dict() if self.ema_model is not None else None,
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "global_step": self.global_step,
                "total_samples": total_samples if total_samples is not None else self.start_total_samples,
                "version": self.version,
                "pytorch_version": torch.__version__,
            }
            torch.save(snapshot, path)

        if should_export:
            export_target = self.get_inference_model()
            export_model_isolated(export_target, pt_path, onnx_path, self.variant, self.device)

        with self._last_ckpt_lock:
            self.metrics["last_ckpt_path"] = path
            self.metrics["last_global_step"] = self.global_step
        print(f"[TR-{self.variant.id}] 💾 checkpoint 已保存/更新: {path} + {pt_path} "
              f"(global_step={self.global_step}, v{self.version}, force={force})")


    def _anneal_value_weight(self, round_idx: int):
        if self.anneal_rounds and self.buffer.cfg.VALUE_TARGET_MODE == "anneal":
            w = min(1.0, (round_idx + 1) / max(self.anneal_rounds, 1))
            self.buffer.value_result_weight = w
            print(f"[TR-{self.variant.id}] value 目标退火 w={w:.3f} (round {round_idx})")

    def _ensure_fixed_eval_from_selfplay(self, samples: List[Dict]) -> None:
        """无归档时，从本会话自对弈原始样本池中按终局结果分层构建固定验证集。"""
        if self._fixed_eval is not None:
            return
        n_fixed = getattr(self.cfg, "VALUE_DRIFT_NUM_POSITIONS", 128)
        if n_fixed <= 0:
            return
        self._raw_sample_pool.extend(samples)
        if len(self._raw_sample_pool) < n_fixed:
            return
        # 防止池子无限累积导致内存增长：设一个上限（n_fixed*2），超过即截断
        max_pool = max(n_fixed * 2, 512)
        if len(self._raw_sample_pool) > max_pool:
            self._raw_sample_pool = self._raw_sample_pool[-max_pool:]
        pool = select_balanced_fixed_samples(self._raw_sample_pool, n_fixed)
        fixed = build_fixed_eval(pool, self.variant) if pool else None
        # 无论构建成功与否都清空池子，避免 build 失败时无界累积
        self._raw_sample_pool = []
        if fixed is not None:
            self._fixed_eval = fixed

    def _permutation(self, transform: str) -> list:
        """获取 Rust 导出的动作置换表（new_policy = old_policy[perm]），带缓存。"""
        perm = self._perm_cache.get(transform)
        if perm is None:
            perm = banqi_4x8.get_action_symmetry_table(
                self.C.BOARD_ROWS, self.C.BOARD_COLS, transform
            )
            self._perm_cache[transform] = perm
        return perm

    def _transform_episode(self, episode_dict: Dict, transform: str) -> Dict:
        """对一个 episode dict 做空间对称增强（全部由 Rust 绑定执行）。"""
        out = dict(episode_dict)
        perm = self._permutation(transform)
        rows, cols = self.C.BOARD_ROWS, self.C.BOARD_COLS
        channels = self.C.TOTAL_INPUT_CHANNELS
        # board 特征空间重排（Rust）
        out["boards"] = [
            banqi_4x8.transform_board(
                list(b), rows, cols, channels, transform
            )
            for b in out["boards"]
        ]
        # policy / action_mask 按置换表 gather（Rust 提供 gather）
        def _gather(p):
            return banqi_4x8.transform_policy(list(p), perm)
        out["policies"] = [_gather(p) for p in out["policies"]]
        out["action_masks"] = [_gather(m) for m in out["action_masks"]]
        if out.get("actions"):
            out["actions"] = [
                int(banqi_4x8.transform_action(a, perm)) for a in out["actions"]
            ]
        return out

    def _maybe_augment(self, episode_dict: Dict) -> List[Dict]:
        """按 config 对 episode 做空间对称增强（动作置换表与 board 重排由 Rust 导出）。

        返回用于训练的 episode dict 列表：
          - DATA_AUGMENT_ENABLED=false：原样返回 [episode_dict]。
          - 开启时：对每局按 DATA_AUGMENT_TRANSFORMS 随机抽一个非恒等变换，
            生成增强副本；DATA_AUGMENT_KEEP_ORIGINAL=true 时保留原始局。
        """
        cfg = self.cfg
        if not getattr(cfg, "DATA_AUGMENT_ENABLED", False):
            return [episode_dict]
        transforms = getattr(cfg, "DATA_AUGMENT_TRANSFORMS", "") or ""
        if transforms:
            transform_list = [
                t.strip() for t in transforms.split(",") if t.strip()
            ]
        else:
            transform_list = list(self.variant.non_identity_transforms)
        # 只保留该变体合法的非恒等变换
        valid = set(self.variant.non_identity_transforms)
        transform_list = [t for t in transform_list if t in valid]
        if not transform_list:
            return [episode_dict]
        keep = getattr(cfg, "DATA_AUGMENT_KEEP_ORIGINAL", True)
        # 每局随机抽 1 个变换（训练侧增强多样性），并保留原始局
        t = transform_list[random.randrange(len(transform_list))]
        out = [episode_dict] if keep else []
        out.append(self._transform_episode(episode_dict, t))
        return out

    def _safe_qsize(self) -> int:
        """线程安全地读取数据队列积压。"""
        try:
            if hasattr(self.data_queue, "qsize"):
                qsize = self.data_queue.qsize()
                return int(qsize) if qsize is not None else -1
            return -1
        except Exception:
            return -1

    def run(self, rounds: int = 100000):
        cfg = self.cfg
        last_processed_round = 0
        version = self.version
        total_samples = self.start_total_samples
        # 批量训练：自对弈数据逐局到达，单局样本量远小于一个合理训练批次。
        # 若每局立即训练，max_batches 会按单局样本量被压到极小，训练碎片化且
        # 反复抽到旧数据。这里累积到足够新样本量（buffer 容量的 1/4）才训练一次，
        # 让训练量充足且聚焦新数据（selfplay 与 rule_selfplay 统一该逻辑）。
        capacity_base = getattr(cfg, "MAX_SAMPLE_BUFFER_SIZE", 50000)
        batch_train_min_samples = max(
            cfg.TRAIN_BATCH * cfg.TRAIN_EPOCHS_PER_ROUND, capacity_base // 4
        )
        pending_samples = 0   # 累积待训练的新样本数
        pending_round = 0     # 累积期间最新的 round_idx
        while not _is_stopped(self.stop_event):
            try:
                episode_dict = self.data_queue.get(timeout=2.0)
            except Exception:
                continue
            if episode_dict is None:
                break

            t0 = time.time()
            # 空间对称增强（动作置换表由 Rust 导出）；关闭时原样返回
            episode_dicts = self._maybe_augment(episode_dict)
            samples: List[Dict] = []
            for ed in episode_dicts:
                samples.extend(episode_to_samples(ed))
            self._ensure_fixed_eval_from_selfplay(samples)
            self.buffer.add_samples(samples)
            new_samples = len(samples)
            total_samples += new_samples
            pending_samples += new_samples
            round_idx = episode_dict.get("round_idx", last_processed_round)
            pending_round = max(pending_round, round_idx)

            min_samples = getattr(cfg, "MIN_SAMPLES_TO_START", getattr(cfg, "TRAIN_MIN_SAMPLES", 100))
            if len(self.buffer) < min_samples:
                print(f"[TR-{self.variant.id}] 等待足够样本进行训练: "
                      f"{len(self.buffer)}/{min_samples}")
                self._maybe_save_early(episode_dict)
                last_processed_round = round_idx
                continue

            # ---- 批量训练门控：累积够新样本才训练，避免单局碎片化训练 ----
            if pending_samples < batch_train_min_samples:
                last_processed_round = round_idx
                continue

            # 本次训练消化 pending_samples 这一批新样本；selfplay 与 rule_selfplay 统一
            new_samples = pending_samples
            pending_samples = 0
            round_idx = pending_round
            self._anneal_value_weight(round_idx)

            # ---- 训练量限制：与累积新增样本量匹配，避免旧数据反复训练 ----
            if new_samples >= capacity_base // 4:
                max_batches = None
            else:
                max_batches = int(
                    (new_samples / cfg.TRAIN_BATCH) * cfg.TRAIN_EPOCHS_PER_ROUND + 0.5
                )
                max_batches = max(max_batches, 1)

            self.model.train()
            epoch_results, total_batches = run_training_epochs(
                self.model, self.optimizer, self.scheduler, self.buffer,
                cfg.TRAIN_EPOCHS_PER_ROUND, self.device, max_batches=max_batches,
                ema_model=self.ema_model if self.ema_enabled else None,
                ema_decay=self.ema_decay,
            )
            self.model.eval()

            self.global_step += total_batches
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.metrics["train_duration"] = time.time() - t0
            self.metrics["global_step"] = self.global_step
            self.metrics["last_lr"] = current_lr
            last_losses = epoch_results[-1] if epoch_results else None
            self.metrics["last_epoch_losses"] = last_losses
            if last_losses is not None:
                with self._stats_lock:
                    self.round_num = round_idx
                    self.total_loss_sum += last_losses[0] * total_batches
                    self.total_policy_loss_sum += last_losses[1] * total_batches
                    self.total_value_loss_sum += last_losses[2] * total_batches
                    self.round_history.append({
                        "round": round_idx,
                        "train_loss": last_losses[0],
                        "policy_loss": last_losses[1],
                        "value_loss": last_losses[2],
                        "grad_norm": last_losses[3],
                        "entropy": last_losses[4],
                        "lr": current_lr,
                        "global_step": self.global_step,
                    })

                print(f"[TR-{self.variant.id}] round {round_idx}: "
                      f"epoch_avg_loss={last_losses[0]:.4f} "
                      f"(policy={last_losses[1]:.4f}, value={last_losses[2]:.4f}) "
                      f"grad_norm={last_losses[3]:.3f} entropy={last_losses[4]:.3f} "
                      f"value_mean={last_losses[5]:.3f} value_std={last_losses[6]:.3f} "
                      f"lr={current_lr:.2e} duration={self.metrics['train_duration']:.1f}s "
                      f"buffer={len(self.buffer)} global_step={self.global_step}")

                # 恢复 TensorBoard 训练过程标量记录
                step = self.global_step
                tag = f"[TR-{self.variant.id}]"
                add_scalar("train/loss", last_losses[0], step)
                add_scalar("train/policy_loss", last_losses[1], step)
                add_scalar("train/value_loss", last_losses[2], step)
                add_scalar("train/grad_norm", last_losses[3], step)
                add_scalar("train/policy_entropy", last_losses[4], step)
                add_scalar("train/value_mean", last_losses[5], step)
                add_scalar("train/value_std", last_losses[6], step)
                add_scalar("train/lr", current_lr, step)
                add_scalar("train/buffer_size", len(self.buffer), step)
                add_scalar("queue/backlog", self._safe_qsize(), step)
                if cfg.VALUE_TARGET_MODE == "anneal":
                    add_scalar("train/value_anneal_w", self.buffer.value_result_weight, step)

                # 固定验证集评估（价值漂移与策略命中率）
                eval_value_drift(self.model, self.device, self._fixed_eval, step, tag, round_idx)
                eval_policy_accuracy(self.model, self.device, self._fixed_eval, step, tag, round_idx)

                # 周期性对战评估
                eval_match_rounds = getattr(cfg, "EVAL_MATCH_ROUNDS", 10)
                if eval_match_rounds > 0 and (round_idx > 0) and (round_idx % eval_match_rounds == 0):
                    eval_match(
                        self.model, self.device, self.variant, cfg,
                        self._prev_weights, round_idx, step, tag
                    )

                # 缓存本轮权重快照（供下一轮 vs prev 对战评估）
                self._prev_weights = {
                    k: v.detach().to("cpu").clone()
                    for k, v in self.model.state_dict().items()
                }

            self.save_checkpoint(new_samples=new_samples, total_samples=total_samples,
                                 round_idx=round_idx)
            last_processed_round = round_idx
            version += 1
            self.version = version
            # 周期内存维护：强制 GC + glibc arena 归还（防 RSS 线性增长）
            self._maintain_memory()

            if last_processed_round >= rounds - 1:
                print(f"[TR-{self.variant.id}] 达到训练轮数上限 {rounds}，退出训练 worker")
                break


    def _maybe_save_early(self, episode_dict):
        # 预热阶段（样本不足）也定期保存，避免长期无 checkpoint
        if (episode_dict.get("round_idx", 0) % 10 == 0) and not os.path.exists(
                self.last_ckpt_path()):
            self.save_checkpoint(round_idx=episode_dict.get("round_idx", 0))

    def _maintain_memory(self, force: bool = False) -> None:
        """周期内存维护：手动 GC + 堆内存空闲页释放。"""
        import gc as _gc
        import ctypes
        self._round_mem_count = getattr(self, "_round_mem_count", 0) + 1
        _gc.collect()
        if (force or self._round_mem_count % 50 == 0) and hasattr(ctypes.CDLL("libc.so.6"), "malloc_trim"):
            ctypes.CDLL("libc.so.6").malloc_trim(0)


    def stats(self) -> Dict[str, float]:
        with self._stats_lock:
            total = max(1, self.global_step)
            return {
                "round_num": self.round_num,
                "total_batches": self.global_step,
                "avg_loss": self.total_loss_sum / total,
                "avg_policy_loss": self.total_policy_loss_sum / total,
                "avg_value_loss": self.total_value_loss_sum / total,
            }

    def round_history_snapshot(self) -> List[Dict]:
        """返回逐轮指标历史的浅拷贝。"""
        with self._stats_lock:
            return list(self.round_history)

    def finalize(self) -> None:
        """优雅退出/结束训练时触发最终 checkpoint 强制保存与导出。"""
        self.save_checkpoint(force=True)
        print(f"[TR-{self.variant.id}] 🎉 最终 Checkpoint 强制保存与导出已完成")

