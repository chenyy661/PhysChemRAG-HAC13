"""紧凑 processed chunk 的只读 PyG 数据集。"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
from torch_geometric.data import Dataset


CHUNK_SIZE = 50_000


class PhysChemRADataset(Dataset):
    """按需读取发布包中的连续模拟样本 chunk。"""

    def __init__(
        self,
        root: str | Path,
        transform=None,
        pre_transform=None,
        max_cached_chunks: int = 3,
        verbose: bool = True,
    ) -> None:
        self.max_cached_chunks = int(max_cached_chunks)
        self.verbose = bool(verbose)
        self._chunk_cache: OrderedDict[int, list] = OrderedDict()
        self._length = 0
        self._chunk_files: list[str] = []
        self._chunk_offsets: list[int] = []
        super().__init__(str(root), transform, pre_transform)
        self._read_manifest()

    @property
    def raw_file_names(self) -> list[str]:
        return []

    @property
    def processed_file_names(self) -> list[str]:
        return ["preprocessing_done.txt"]

    def download(self) -> None:
        return None

    def process(self) -> None:
        raise RuntimeError("发布版只读取已打包数据，不支持从 Parquet 现场预处理")

    def _read_manifest(self) -> None:
        marker = Path(self.processed_dir) / "preprocessing_done.txt"
        if not marker.is_file():
            raise FileNotFoundError(f"缺少模拟数据标记: {marker}")
        fields = dict(part.split("=", 1) for part in marker.read_text().strip().split(","))
        self._length = int(fields["total"])
        chunks = int(fields["chunks"])
        self._chunk_files = [f"chunk_{index}.pt" for index in range(chunks)]
        self._chunk_offsets = [index * CHUNK_SIZE for index in range(chunks)]

    def len(self) -> int:
        return self._length

    def get(self, index: int):
        if not 0 <= int(index) < self._length:
            raise IndexError(index)
        chunk_index = int(index) // CHUNK_SIZE
        local_index = int(index) % CHUNK_SIZE
        if chunk_index not in self._chunk_cache:
            path = Path(self.processed_dir) / self._chunk_files[chunk_index]
            self._chunk_cache[chunk_index] = torch.load(
                path, map_location="cpu", weights_only=False
            )
            if len(self._chunk_cache) > self.max_cached_chunks:
                self._chunk_cache.popitem(last=False)
            if self.verbose:
                print(f"加载 {path.name}")
        else:
            self._chunk_cache.move_to_end(chunk_index)
        return self._chunk_cache[chunk_index][local_index]

