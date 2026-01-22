# CI / Jenkins

There is a `Jenkinsfile` in the repo root; this file contains basic CI/CD steps in Jenkins declarative pipeline format.

What the pipeline does

- Checks out the repo
- Sets up Python virtual environment and installs dependencies
- Runs lint (flake8) and tests (pytest)
- Builds Docker images for `api`, `bot` and `optimizer`
- Optionally pushes images to registry and deploys via SSH

Required Jenkins Credentials (add in Jenkins -> Credentials):

- `docker-registry-creds` — Docker registry username/password
- `deploy-ssh-key` — SSH private key for deployment

Important environment variables (set in job or global env):

- `REGISTRY` — optional registry prefix, e.g. `myregistry.example.com/algo_trading_lab/`
- `PUSH_IMAGES=true` — to push images
- `DEPLOY=true` — to run deploy step
- `DEPLOY_HOST` and `DEPLOY_DIR` — deployment host and directory

Quick test (local):

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install pytest flake8
flake8 . || true
pytest -q
```

Notes:

- If your Jenkins agent cannot do Docker builds, use Kaniko/BuildKit alternatives.
- Store credentials in Jenkins Credentials; do not use plain text in pipeline.
