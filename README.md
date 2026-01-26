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
CUDA_VISIBLE_DEVICES=0 python -m src.yaku.exp1.training.train --yaku_indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32 --gpu 0 --no_wandb
```
