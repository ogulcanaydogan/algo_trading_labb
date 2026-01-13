# CI / Jenkins

A `Jenkinsfile` is located in the repo root; this file contains basic CI/CD steps as a Jenkins declarative pipeline.

## What the Pipeline Does

- Checks out the repository
- Sets up Python virtual environment and installs dependencies
- Runs lint (flake8) and tests (pytest)
- Builds Docker images for `api`, `bot`, and `optimizer`
- Optionally pushes images to registry and deploys via SSH

## Required Jenkins Credentials

Add these in Jenkins -> Credentials:

- `docker-registry-creds` — Docker registry username/password
- `deploy-ssh-key` — SSH private key for deployment

## Important Environment Variables

Set these in job or global env:

- `REGISTRY` — Optional registry prefix, e.g., `myregistry.example.com/algo_trading_lab/`
- `PUSH_IMAGES=true` — To push images
- `DEPLOY=true` — To run the deploy step
- `DEPLOY_HOST` and `DEPLOY_DIR` — Host and directory for deployment

## Quick Local Test

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install pytest flake8
flake8 . || true
pytest -q
```

## Notes

- If your Jenkins agent can't do Docker builds, use Kaniko/BuildKit alternatives.
- Store credentials in Jenkins Credentials; don't use plain text in pipeline.
