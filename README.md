# Mahjong Imitator

## Environment Setup

### 1. Configure Environment Variables
Create your local environment file from the example, then set your user ID and group ID, secure password.
```
cp .env.example .env
```

### 2. Run Docker Containers
Run the application and database containers.
```
docker compose up -d --build
docker compose exec app bash
```