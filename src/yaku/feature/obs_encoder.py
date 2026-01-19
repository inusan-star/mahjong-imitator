import mjx
import numpy as np

from src.yaku import config as yaku_config


class ObservationEncoder:
    """Encoder for mahjong game observations into feature."""

    def encode(self, obs: mjx.Observation) -> np.ndarray:
        """Encode observation into feature."""
        feature = np.zeros(yaku_config.INPUT_DIM, dtype=np.int32)

        feature[0:1] = self._encode_seat_number(obs)
        feature[1:35] = self._encode_hands(obs)
        feature[35:171] = self._encode_discards(obs)
        feature[171:427] = self._encode_melds(obs)
        feature[427:461] = self._encode_dora_indicators(obs)
        feature[461:462] = self._encode_prevailing_wind(obs)
        feature[462:463] = self._encode_own_wind(obs)
        feature[463:467] = self._encode_richi_players_last_round(obs)
        feature[467:471] = self._encode_richi_players(obs)

        return feature

    def _encode_seat_number(self, obs: mjx.Observation) -> np.ndarray:
        """Encode seat number."""
        feature = np.zeros(1, dtype=np.int32)

        feature[0] = obs.who()

        return feature

    def _encode_hands(self, obs: mjx.Observation) -> np.ndarray:
        """Encode hands."""
        feature = np.zeros(34, dtype=np.int32)

        closed_tiles = obs.curr_hand().closed_tiles()

        for tile in closed_tiles:
            feature[tile.type()] += 1

        return feature

    def _encode_discards(self, obs: mjx.Observation) -> np.ndarray:
        """Encode discards."""
        feature = np.zeros(136, dtype=np.int32)

        for event in obs.events():
            if event.type() in [mjx.EventType.DISCARD, mjx.EventType.TSUMOGIRI]:
                feature[event.who() * 34 + event.tile().type()] += 1

        return feature

    def _encode_melds(self, obs: mjx.Observation) -> np.ndarray:
        """Encode melds."""
        feature = np.zeros(256, dtype=np.int32)

        player_codes: list[list[int]] = [[] for _ in range(4)]
        player_meld_tile_sets: list[list[list[int]]] = [[] for _ in range(4)]

        for event in obs.events():
            player = event.who()
            event_type = event.type()

            if event_type in [mjx.EventType.CHI, mjx.EventType.PON, mjx.EventType.OPEN_KAN, mjx.EventType.CLOSED_KAN]:
                if len(player_codes[player]) < 4:
                    player_codes[player].append(event.open().bit)
                    player_meld_tile_sets[player].append([t.type() for t in event.open().tiles()])

            elif event_type == mjx.EventType.ADDED_KAN:
                added_tile_type = event.open().last_tile().type()

                for i, meld_tile_set in enumerate(player_meld_tile_sets[player]):
                    if all(t == added_tile_type for t in meld_tile_set):
                        player_codes[player][i] = event.open().bit
                        player_meld_tile_sets[player][i].append(added_tile_type)
                        break

        for player in range(4):
            for meld_idx, code in enumerate(player_codes[player]):
                for bit in range(16):
                    feature[player * 64 + meld_idx * 16 + bit] = (code >> bit) & 1

        return feature

    def _encode_dora_indicators(self, obs: mjx.Observation) -> np.ndarray:
        """Encode dora indicators."""
        feature = np.zeros(34, dtype=np.int32)

        for _, dora in enumerate(obs.doras()):
            if dora < 27:
                if dora % 9 == 0:
                    indicator = dora + 8

                else:
                    indicator = dora - 1

            elif dora < 31:
                if dora == 27:
                    indicator = 30

                else:
                    indicator = dora - 1

            else:
                if dora == 31:
                    indicator = 33

                else:
                    indicator = dora - 1

            feature[indicator] += 1

        return feature

    def _encode_prevailing_wind(self, obs: mjx.Observation) -> np.ndarray:
        """Encode prevailing wind."""

        feature = np.zeros(1, dtype=np.int32)

        feature[0] = obs.round() // 4

        return feature

    def _encode_own_wind(self, obs: mjx.Observation) -> np.ndarray:
        """Encode own wind."""
        feature = np.zeros(1, dtype=np.int32)

        feature[0] = (obs.who() - obs.dealer() + 4) % 4

        return feature

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
