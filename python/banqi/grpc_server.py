# python/banqi/grpc_server.py
"""
gRPC 自对弈服务端（自对弈 / 训练分离架构中的「训练端」）。

训练端同时扮演两种 gRPC 角色：
  1. **Server**（监听本机，供 Rust Worker 连接）：
     - `ReportGameMeta`：接收 Worker 上报的样本元信息（含 data_id），记录待拉取。
     - `FetchLatestModel`：向 Worker 提供最新 TorchScript 模型文件流（供其热更新）。
     - `SyncControl`：向 Worker 下发控制命令（MCTS 模拟次数、探索温度等）。
  2. **Client**（连接 Rust Worker 的 serve 端口）：
     - `PullGameData`：按 data_id 主动拉取样本数据流，反序列化后写入训练队列。

训练端不依赖 Rust PyO3 绑定（banqi_4x8），可独立启动（python -m banqi.grpc_server），
仅需 grpcio / protobuf / pyyaml 即可提供模型与控制在线上服务。
"""

import json
import os
import time
import logging
import threading
from concurrent import futures

import grpc

from .proto import pb, pb_grpc

logger = logging.getLogger("banqi.grpc_server")


class BanqiSelfPlayServicer(pb_grpc.SelfPlayServiceServicer):
    """gRPC Server 端实现：ReportGameMeta / FetchLatestModel / SyncControl。"""

    def __init__(self, data_queue=None, model_path_provider=None, config_provider=None):
        """
        :param data_queue: 训练数据队列（PullGameData 拉到的样本写入这里）
        :param model_path_provider: Callable，返回当前最新 TorchScript 模型文件路径
        :param config_provider: Callable，返回当前控制参数（mcts_sims 等）
        """
        self.data_queue = data_queue
        self.model_path_provider = model_path_provider
        self.config_provider = config_provider

        # Worker 上报的待拉取样本元信息：data_id -> meta dict
        self.pending_metas: dict = {}
        self.lock = threading.Lock()

    # ---- 1. 接收元信息，记录待拉取 data_id ----
    def ReportGameMeta(self, request, context):
        logger.info(
            "收到 Worker [%s] 元信息上报: data_id=%s, games=%s, steps=%s",
            request.worker_id, request.data_id, request.game_count, request.total_steps,
        )
        with self.lock:
            self.pending_metas[request.data_id] = {
                "worker_id": request.worker_id,
                "game_count": request.game_count,
                "total_steps": request.total_steps,
                "timestamp": request.timestamp,
                "model_version": request.model_version,
            }
        return pb.ReportMetaResponse(
            accepted=True,
            message="Meta received, waiting for pull",
        )

    # ---- 2. 提供模型文件流 ----
    def FetchLatestModel(self, request, context):
        model_path = None
        if self.model_path_provider:
            model_path = self.model_path_provider()

        if not model_path or not os.path.exists(model_path):
            logger.warning("模型拉取请求失败: 模型路径不存在 %s", model_path)
            return

        version_str = str(os.path.getmtime(model_path))
        chunk_size = 64 * 1024  # 64KB per chunk

        with open(model_path, "rb") as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                is_last = len(data) < chunk_size
                yield pb.ModelChunk(
                    version=version_str,
                    chunk_data=data,
                    is_last=is_last,
                )

    # ---- 3. 下发控制命令 ----
    def SyncControl(self, request, context):
        mcts_sims = 128
        temperature = 1.0
        playout_cap_random = False
        pause = False

        if self.config_provider:
            cfg = self.config_provider()
            mcts_sims = cfg.get("mcts_sims", 128)
            temperature = cfg.get("temperature", 1.0)
            playout_cap_random = cfg.get("playout_cap_random", False)

        return pb.ControlCommand(
            mcts_sims=mcts_sims,
            temperature=temperature,
            playout_cap_random=playout_cap_random,
            pause_self_play=pause,
        )

    # ---- 供 Puller 消费的待拉取队列接口 ----
    def take_pending_data_id(self) -> str:
        """取出一个待拉取 data_id（FIFO）。无则返回 None。"""
        with self.lock:
            for data_id in self.pending_metas:
                del self.pending_metas[data_id]
                return data_id
        return None

    def requeue_data_id(self, data_id: str) -> None:
        """拉取失败时放回待拉取队列（线程安全）。"""
        with self.lock:
            if data_id not in self.pending_metas:
                # 保持 FIFO 语义：放到队尾
                new_metas = {k: v for k, v in self.pending_metas.items()}
                self.pending_metas.clear()
                self.pending_metas.update(new_metas)
                self.pending_metas[data_id] = {}
            else:
                self.pending_metas[data_id] = self.pending_metas.get(data_id, {})


class WorkerPuller(threading.Thread):
    """gRPC Client 端：连接 Rust Worker，按 data_id 主动拉取样本并写入训练队列。"""

    def __init__(self, worker_host, worker_port, servicer, data_queue,
                 poll_interval=0.5, round_counter=None):
        super().__init__(name="GrpcWorkerPuller", daemon=True)
        self.address = f"{worker_host}:{worker_port}"
        self.servicer = servicer
        self.data_queue = data_queue
        self.poll_interval = poll_interval
        self._round_counter = round_counter if round_counter is not None else iter_count()
        self.pulled_games = 0
        self.pulled_samples = 0

    def run(self):
        logger.info("🔄 Worker 数据拉取线程连接 %s", self.address)
        while True:
            data_id = self.servicer.take_pending_data_id()
            if data_id is None:
                time.sleep(self.poll_interval)
                continue
            self._pull(data_id)

    def _pull(self, data_id):
        try:
            with grpc.insecure_channel(self.address) as channel:
                stub = pb_grpc.SelfPlayServiceStub(channel)
                resp = stub.PullGameData(pb.PullDataRequest(data_id=data_id))
                payload = b"".join(chunk.payload for chunk in resp)
        except Exception as exc:  # noqa: BLE001 - 网络波动时把 data_id 放回，稍后重试
            logger.warning("拉取 data_id=%s 失败: %s（放回重试）", data_id, exc)
            self.servicer.requeue_data_id(data_id)
            time.sleep(self.poll_interval)
            return

        if not payload:
            logger.debug("data_id=%s 无有效 payload（可能是空批次）", data_id)
            return

        try:
            episodes = json.loads(payload.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            logger.warning("反序列化 data_id=%s 失败: %s", data_id, exc)
            return

        round_idx = next(self._round_counter)
        for ep in episodes:
            ep["round_idx"] = round_idx
            try:
                self.data_queue.put(ep)
            except Exception:  # noqa: BLE001 - 队列满/关闭时丢弃
                logger.warning("训练队列写入失败，丢弃 data_id=%s 的一局", data_id)
                continue
            self.pulled_games += 1
            self.pulled_samples += len(ep.get("boards", []))
        logger.info(
            "✅ 已拉取 data_id=%s（%d 局，%d 样本）并写入训练队列",
            data_id, len(episodes), self.pulled_samples,
        )


def iter_count():
    """自增 round_idx 生成器。"""
    n = 0
    while True:
        n += 1
        yield n


class GrpcServerThread(threading.Thread):
    """后台 gRPC 服务端线程（提供模型 / 控制命令，接收元信息），并联动 Puller 拉取样本。"""

    def __init__(self, host="0.0.0.0", port=50051, max_workers=10,
                 worker_host=None, worker_port=None, pull_enabled=True,
                 data_queue=None, model_path_provider=None, config_provider=None):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.max_workers = max_workers
        self.server = None
        self.servicer = BanqiSelfPlayServicer(
            data_queue=data_queue,
            model_path_provider=model_path_provider,
            config_provider=config_provider,
        )
        self.puller = None
        # 若配置了 worker 地址，启动样本拉取线程
        if pull_enabled and worker_host and data_queue is not None:
            self.puller = WorkerPuller(
                worker_host=worker_host,
                worker_port=worker_port or 50052,
                servicer=self.servicer,
                data_queue=data_queue,
            )

    def run(self):
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=self.max_workers))
        pb_grpc.add_SelfPlayServiceServicer_to_server(self.servicer, self.server)
        address = f"{self.host}:{self.port}"
        self.server.add_insecure_port(address)
        self.server.start()
        logger.info("✅ gRPC 自对弈服务端已在 %s 启动 (后台线程)", address)
        if self.puller is not None:
            self.puller.start()
            logger.info("✅ 样本拉取线程已启动")
        self.server.wait_for_termination()

    def stop(self):
        if self.server:
            self.server.stop(grace=2.0)
            logger.info("🛑 gRPC 自对弈服务端已关闭")


# =============================================================================
# 独立启动入口：python -m banqi.grpc_server
# 不依赖 Rust PyO3 绑定，仅用于提供模型 / 控制命令在线服务。
# =============================================================================
def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Banqi gRPC 自对弈训练端服务")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=50051, help="监听端口")
    parser.add_argument("--model", default="", help="TorchScript 模型路径 (.pt)")
    parser.add_argument("--mcts-sims", type=int, default=128, help="下发 MCTS 模拟次数")
    parser.add_argument("--max-workers", type=int, default=10, help="服务端并发工作线程数")
    args = parser.parse_args()

    model_path = args.model

    def model_path_provider():
        return model_path if os.path.exists(model_path) else None

    def config_provider():
        return {"mcts_sims": args.mcts_sims, "temperature": 1.0, "playout_cap_random": False}

    thread = GrpcServerThread(
        host=args.host, port=args.port, max_workers=args.max_workers,
        pull_enabled=False,
        model_path_provider=model_path_provider,
        config_provider=config_provider,
    )
    thread.start()
    logger.info("独立训练端服务已启动（不拉取样本，仅服务模型与控制）")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        thread.stop()


if __name__ == "__main__":
    main()
