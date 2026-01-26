import duckdb
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Union

import src.config as global_config
import src.yaku.common.config as common_config


class YakuEncoder:
    """Encoder for mahjong yaku names using MJX IDs."""

    def __init__(self) -> None:
        """Initialize the encoder."""
        if not common_config.YAKU_DISTRIBUTION_FILE.exists():
            raise FileNotFoundError(f"Distribution file not found: {common_config.YAKU_DISTRIBUTION_FILE}")

        yaku_distribution_df = pd.read_csv(common_config.YAKU_DISTRIBUTION_FILE)
        yaku_distribution_df = yaku_distribution_df[yaku_distribution_df["Yaku Name"].notna()]

        self.yaku_names = yaku_distribution_df["Yaku Name"].tolist()
        self.name_to_index = {name: i for i, name in enumerate(self.yaku_names)}
        self.index_to_name = {i: name for i, name in enumerate(self.yaku_names)}
        self.num_classes = len(self.yaku_names)
        self.mjx_id_to_index = self._load_mjx_id_to_index()

    def _load_mjx_id_to_index(self) -> Dict[int, int]:
        """Load MJX yaku ID to internal multi-hot index mapping."""
        yaku_parquet_path = f"{global_config.DUMPS_DIR}/yaku/*.parquet"

        connection = duckdb.connect()
        query = f"SELECT id, name FROM read_parquet('{yaku_parquet_path}')"
        yaku_master_df = connection.execute(query).df()
        connection.close()

        mapping = {}

        for row in yaku_master_df.itertuples(index=False):
            if row.name in self.name_to_index:
                mapping[int(row.id) - 1] = self.name_to_index[row.name]

        return mapping

    def encode(self, yaku_id_list: List[int]) -> torch.Tensor:
        """Encode MJX yaku IDs into a multi-hot vector."""
        multi_hot_vector = torch.zeros(self.num_classes)

        for yaku_id in yaku_id_list:
            if yaku_id in self.mjx_id_to_index:
                multi_hot_vector[self.mjx_id_to_index[yaku_id]] = 1.0

        return multi_hot_vector

    def decode(self, vector: Union[torch.Tensor, np.ndarray], threshold: float = 0.5) -> List[str]:
        """Decode a vector into yaku names."""
        if isinstance(vector, torch.Tensor):
            vector = vector.detach().cpu().numpy()

        detected_indices = np.where(vector >= threshold)[0]
        return [self.index_to_name[int(index)] for index in detected_indices]

    @property
    def labels(self) -> List[str]:
        """Get the list of yaku names (Labels)."""
        return self.yaku_names
