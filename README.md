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
python -m src.yaku.exp1.feature.create_dataset
python -m src.yaku.exp1.training.train
```
