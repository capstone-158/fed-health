# How to run
Prerequisites:
```bash
brew install --cask orbstack
brew install docker

# In the main project directory
mkdir data/ models/
git pull origin main
```

for macos:
```bash
orb start
docker compose up --build
```

## Running the Project After the Initial Build

Once the project has been built successfully for the first time, you do not need to rebuild it every time.

Just do:

```bash
orb start
docker compose up -d
```

If `Dockerfile` or `requirements.txt` is modified, then rebuild the images using:

```bash
docker compose up --build
```

todo: check TODO marker
