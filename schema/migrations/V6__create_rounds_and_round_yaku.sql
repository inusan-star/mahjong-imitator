CREATE TABLE IF NOT EXISTS `rounds` (
    `id` INT NOT NULL AUTO_INCREMENT,
    `game_id` INT NOT NULL,
    `round_index` INT NOT NULL,
    `honba` INT NOT NULL DEFAULT 0,
    `is_agari` BOOLEAN NOT NULL DEFAULT FALSE,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    FOREIGN KEY (`game_id`) REFERENCES `games` (`id`),
    UNIQUE KEY `uk_game_round_honba` (`game_id`, `round_index`, `honba`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `round_yaku` (
    `id` INT NOT NULL AUTO_INCREMENT,
    `round_id` INT NOT NULL,
    `yaku_id` INT NOT NULL,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    FOREIGN KEY (`round_id`) REFERENCES `rounds` (`id`),
    FOREIGN KEY (`yaku_id`) REFERENCES `yaku` (`id`),
    UNIQUE KEY `uk_round_yaku` (`round_id`, `yaku_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;