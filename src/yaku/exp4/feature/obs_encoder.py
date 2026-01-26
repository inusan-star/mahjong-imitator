import mjx
import numpy as np
import src.yaku.exp4.config as exp4_config


class ObservationEncoder:
    """Encoder for mahjong game observations into a 35xN feature matrix for Transformer."""

    def encode(self, obs: mjx.Observation) -> np.ndarray:
        """
        Encode observation into feature matrix.
        Returns: np.ndarray of shape (35, feature_dim)
        - 0-33: Tile Tokens (one for each of the 34 tile types)
        - 34: Global Token (contextual information)
        """
        # 各牌トークンの特徴量 (34, dim) を収集
        tile_hands = self._encode_hands(obs)  # (34, 1)
        tile_discards = self._encode_discards(obs)  # (34, 4) - 各プレイヤーの河
        tile_melds = self._encode_melds(obs)  # (34, 16) - 各プレイヤーの副露(4人×4メンツ)
        tile_dora = self._encode_dora_indicators(obs)  # (34, 1)

        # 34個の牌トークンを結合 (34, 22)
        tile_features = np.concatenate([tile_hands, tile_discards, tile_melds, tile_dora], axis=1)

        # グローバル情報の収集
        seat = self._encode_seat_number(obs)  # (1,)
        p_wind = self._encode_prevailing_wind(obs)  # (1,)
        o_wind = self._encode_own_wind(obs)  # (1,)
        r_last = self._encode_richi_players_last_round(obs)  # (4,)
        r_now = self._encode_richi_players(obs)  # (4,)

        # グローバル情報を結合して1つのベクトルに (11,)
        global_vector = np.concatenate([seat, p_wind, o_wind, r_last, r_now])

        # 牌トークンの次元数に合わせるためパディング (11 -> 22)
        target_dim = tile_features.shape[1]

        if len(global_vector) > target_dim:
            global_token = global_vector[:target_dim]  # 切り詰め
        else:
            global_token = np.pad(global_vector, (0, target_dim - len(global_vector)))

        # 35番目のトークンとして結合 (35, 22)
        feature = np.vstack([tile_features, global_token.reshape(1, -1)])

        return feature.astype(np.float32)

    def _encode_seat_number(self, obs: mjx.Observation) -> np.ndarray:
        return np.array([obs.who()], dtype=np.int32)

    def _encode_hands(self, obs: mjx.Observation) -> np.ndarray:
        feature = np.zeros((34, 1), dtype=np.int32)
        for tile in obs.curr_hand().closed_tiles():
            feature[tile.type(), 0] += 1
        return feature

    def _encode_discards(self, obs: mjx.Observation) -> np.ndarray:
        feature = np.zeros((34, 4), dtype=np.int32)
        for event in obs.events():
            if event.type() in [mjx.EventType.DISCARD, mjx.EventType.TSUMOGIRI]:
                feature[event.tile().type(), event.who()] += 1
        return feature

    def _encode_melds(self, obs: mjx.Observation) -> np.ndarray:
        """Encode melds into a 34x16 matrix (4 players * 4 melds per player)."""
        # 各プレイヤー4列ずつ、計16列を確保 [cite: 159]
        feature = np.zeros((34, 16), dtype=np.int32)
        player_meld_count = [0] * 4
        # 加槓（ADDED_KAN）の処理用に、どの牌種がどの副露インデックスにあるかを保持
        meld_map = {}  # (player, tile_type) -> meld_index

        for event in obs.events():
            p = event.who()
            e_type = event.type()

            # 新規の副露（チー、ポン、大明槓、暗槓） [cite: 186]
            if e_type in [mjx.EventType.CHI, mjx.EventType.PON, mjx.EventType.OPEN_KAN, mjx.EventType.CLOSED_KAN]:
                if player_meld_count[p] < 4:  # 最大4回まで [cite: 158]
                    m_idx = player_meld_count[p]
                    for tile in event.open().tiles():
                        t_type = tile.type()
                        feature[t_type, p * 4 + m_idx] += 1
                        meld_map[(p, t_type)] = m_idx
                    player_meld_count[p] += 1

            # 加槓：既存のポンに牌を追加する
            elif e_type == mjx.EventType.ADDED_KAN:
                t_type = event.open().last_tile().type()
                if (p, t_type) in meld_map:
                    m_idx = meld_map[(p, t_type)]
                    feature[t_type, p * 4 + m_idx] += 1

        return feature

    def _encode_dora_indicators(self, obs: mjx.Observation) -> np.ndarray:
        feature = np.zeros((34, 1), dtype=np.int32)
        for dora in obs.doras():
            feature[dora, 0] += 1
        return feature

    def _encode_prevailing_wind(self, obs: mjx.Observation) -> np.ndarray:
        return np.array([obs.round() // 4], dtype=np.int32)

    def _encode_own_wind(self, obs: mjx.Observation) -> np.ndarray:
        return np.array([(obs.who() - obs.dealer() + 4) % 4], dtype=np.int32)

    def _encode_richi_players_last_round(self, obs: mjx.Observation) -> np.ndarray:
        feature = np.zeros(4, dtype=np.int32)
        events = obs.events()
        for i, event in enumerate(events):
            if event.type() == mjx.EventType.RIICHI:
                player = event.who()
                is_last_riichi = True
                for later_event in events[i + 2 :]:
                    if later_event.type() in [
                        mjx.EventType.CHI,
                        mjx.EventType.PON,
                        mjx.EventType.OPEN_KAN,
                        mjx.EventType.CLOSED_KAN,
                        mjx.EventType.ADDED_KAN,
                    ]:
                        is_last_riichi = False
                        break
                    if later_event.who() == player and later_event.type() in [
                        mjx.EventType.DISCARD,
                        mjx.EventType.TSUMOGIRI,
                    ]:
                        is_last_riichi = False
                        break
                if is_last_riichi:
                    feature[player] = 1
        return feature

    def _encode_richi_players(self, obs: mjx.Observation) -> np.ndarray:
        feature = np.zeros(4, dtype=np.int32)
        for event in obs.events():
            if event.type() == mjx.EventType.RIICHI:
                feature[event.who()] = 1
        return feature
