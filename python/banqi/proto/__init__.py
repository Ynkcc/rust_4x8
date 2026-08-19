# python/banqi/proto/__init__.py
"""
banqi.proto 独立 Protobuf/gRPC 模块包
"""

from . import banqi_service_pb2 as pb
from . import banqi_service_pb2_grpc as pb_grpc

__all__ = ["pb", "pb_grpc"]
