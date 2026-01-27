import mjx
import numpy as np


class ObservationEncoder:
    """Encoder for mahjong game observations into feature."""

    def encode(self, obs: mjx.Observation) -> np.ndarray:
        """Encode observation into feature."""
        tile_hands = self._encode_hands(obs)
        tile_discards = self._encode_discards(obs)
        tile_melds = self._encode_melds(obs)
        tile_dora = self._encode_dora_indicators(obs)

        tile_features = np.concatenate([tile_hands, tile_discards, tile_melds, tile_dora], axis=1)

        seat_number = self._encode_seat_number(obs)
        prevailing_wind = self._encode_prevailing_wind(obs)
        own_wind = self._encode_own_wind(obs)
        richi_players_last_round = self._encode_richi_players_last_round(obs)
        richi_players = self._encode_richi_players(obs)

        global_vector = np.concatenate(
            [seat_number, prevailing_wind, own_wind, richi_players_last_round, richi_players]
        )

        target_dimension = tile_features.shape[1]
        global_token = np.pad(global_vector, (0, target_dimension - len(global_vector)))

        feature = np.vstack([tile_features, global_token.reshape(1, -1)])

        return feature.astype(np.float32)  # (35, 22)

    def _encode_seat_number(self, obs: mjx.Observation) -> np.ndarray:
        """Encode seat number."""
        return np.array([obs.who()], dtype=np.int32)

    def _encode_hands(self, obs: mjx.Observation) -> np.ndarray:
        """Encode hands."""
        feature = np.zeros((34, 1), dtype=np.int32)

        for tile in obs.curr_hand().closed_tiles():
            feature[tile.type(), 0] += 1

        return feature

    def _encode_discards(self, obs: mjx.Observation) -> np.ndarray:
        """Encode discards."""
        feature = np.zeros((34, 4), dtype=np.int32)

        for event in obs.events():
            if event.type() in [mjx.EventType.DISCARD, mjx.EventType.TSUMOGIRI]:
                feature[event.tile().type(), event.who()] += 1

        return feature

    def _encode_melds(self, obs: mjx.Observation) -> np.ndarray:
        """Encode melds."""
        feature = np.zeros((34, 16), dtype=np.int32)
        player_meld_count = [0] * 4
        meld_map = {}

        for event in obs.events():
            player = event.who()
            event_type = event.type()

            if event_type in [mjx.EventType.CHI, mjx.EventType.PON, mjx.EventType.OPEN_KAN, mjx.EventType.CLOSED_KAN]:
                if player_meld_count[player] < 4:
                    meld_index = player_meld_count[player]
                    for tile in event.open().tiles():
                        tile_type = tile.type()
                        feature[tile_type, player * 4 + meld_index] += 1
                        meld_map[(player, tile_type)] = meld_index
                    player_meld_count[player] += 1

            elif event_type == mjx.EventType.ADDED_KAN:
                tile_type = event.open().last_tile().type()
                if (player, tile_type) in meld_map:
                    meld_index = meld_map[(player, tile_type)]
                    feature[tile_type, player * 4 + meld_index] += 1

        return feature

    def _encode_dora_indicators(self, obs: mjx.Observation) -> np.ndarray:
        """Encode dora indicators."""
        feature = np.zeros((34, 1), dtype=np.int32)

        for dora in obs.doras():
            tile_type = dora

            if tile_type < 27:
                if tile_type % 9 == 0:
                    indicator = tile_type + 8
                else:
                    indicator = tile_type - 1
            elif tile_type < 31:
                if tile_type == 27:
                    indicator = 30
                else:
                    indicator = tile_type - 1
            else:
                if tile_type == 31:
                    indicator = 33
                else:
                    indicator = tile_type - 1
            feature[indicator, 0] += 1

        return feature

    def _encode_prevailing_wind(self, obs: mjx.Observation) -> np.ndarray:
        """Encode prevailing wind."""
        return np.array([obs.round() // 4], dtype=np.int32)

    def _encode_own_wind(self, obs: mjx.Observation) -> np.ndarray:
        """Encode own wind."""
        return np.array([(obs.who() - obs.dealer() + 4) % 4], dtype=np.int32)

    def _encode_richi_players_last_round(self, obs: mjx.Observation) -> np.ndarray:
        """Encode richi players last round."""
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
        """Encode richi players."""
        feature = np.zeros(4, dtype=np.int32)

        for event in obs.events():
            if event.type() == mjx.EventType.RIICHI:
                feature[event.who()] = 1

        return feature
