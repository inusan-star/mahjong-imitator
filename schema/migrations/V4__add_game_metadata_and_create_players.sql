ALTER TABLE games
    ADD COLUMN game_type INT NOT NULL AFTER log_id,
    ADD COLUMN lobby_id INT NOT NULL AFTER game_type;

CREATE TABLE IF NOT EXISTS `players` (
    `id` INT NOT NULL AUTO_INCREMENT,
    `name` VARCHAR(255) NOT NULL UNIQUE,
    `gender` VARCHAR(10) NOT NULL,
    `last_dan` INT NOT NULL,
    `last_rate` FLOAT NOT NULL,
    `max_dan` INT NOT NULL,
    `max_rate` FLOAT NOT NULL,
    `last_played_at` DATETIME NOT NULL,
    `game_count` INT NOT NULL DEFAULT 0,
    `first_count` INT NOT NULL DEFAULT 0,
    `second_count` INT NOT NULL DEFAULT 0,
    `third_count` INT NOT NULL DEFAULT 0,
    `fourth_count` INT NOT NULL DEFAULT 0,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `game_players` (
    `id` INT NOT NULL AUTO_INCREMENT,
    `game_id` INT NOT NULL,
    `player_id` INT NOT NULL,
    `seat_index` INT NOT NULL,
    `dan` INT NOT NULL,
    `rate` FLOAT NOT NULL,
    `score` INT NOT NULL,
    `rank` INT NOT NULL,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    FOREIGN KEY (`game_id`) REFERENCES `games` (`id`),
    FOREIGN KEY (`player_id`) REFERENCES `players` (`id`),
    UNIQUE KEY `uk_game_player` (`game_id`, `player_id`),
    UNIQUE KEY `uk_game_seat` (`game_id`, `seat_index`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;