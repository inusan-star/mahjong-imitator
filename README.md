# Mahjong Imitator

## Environment Setup

### 1. Configure Environment Variables

Create your local environment file from the example.

```
cp .env.example .env
```

### 2. Run Docker Containers

Run the application and database containers.

```
docker compose up -d --build
docker compose exec app bash
```

### 3. Commands

```
python -m src.data.main
python -m src.data.export
python -m src.yaku.common.split_dataset
python -m src.yaku.exp1.feature.create_dataset
CUDA_VISIBLE_DEVICES=GPU-c1b713ce-d6a4-e657-727a-32f16a581f53 python -m src.yaku.exp1.training.train --no_wandb
CUDA_VISIBLE_DEVICES=GPU-c1b713ce-d6a4-e657-727a-32f16a581f53 python -m src.yaku.exp1.training.test
CUDA_VISIBLE_DEVICES=GPU-c1b713ce-d6a4-e657-727a-32f16a581f53 python -m src.yaku.exp2.training.train --no_wandb
CUDA_VISIBLE_DEVICES=GPU-c1b713ce-d6a4-e657-727a-32f16a581f53 python -m src.yaku.exp2.training.test
CUDA_VISIBLE_DEVICES=GPU-c1b713ce-d6a4-e657-727a-32f16a581f53 python -m src.yaku.exp3.training.train --no_wandb
CUDA_VISIBLE_DEVICES=GPU-c1b713ce-d6a4-e657-727a-32f16a581f53 python -m src.yaku.exp3.training.test
python -m src.yaku.exp4.feature.create_dataset
```
