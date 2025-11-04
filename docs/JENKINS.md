# CI with Jenkins

This repository includes a `Jenkinsfile` at the project root that provides a declarative pipeline suitable for typical CI/CD flows.

What the pipeline does

- Checkout the repository
- Set up Python virtual environment and install dependencies
- Run linting (flake8) and tests (pytest)
- Build Docker images for `api`, `bot` and `optimizer` using the repository `Dockerfile`
- Optionally push images to a registry and deploy via SSH (controlled by pipeline env vars)

Required Jenkins credentials (add these in Jenkins -> Credentials):

- `docker-registry-creds` — Username & password for Docker registry
- `deploy-ssh-key` — SSH private key used for deploying to remote host

Important environment variables (set in the job or globally):

- `REGISTRY` — optional registry prefix, e.g. `myregistry.example.com/algo_trading_lab/`
- `PUSH_IMAGES=true` — push built images when set
- `DEPLOY=true` — run the deploy stage when set
- `DEPLOY_HOST` / `DEPLOY_DIR` — deployment host and directory for `docker compose up -d`

Quick tips

- If your Jenkins agents cannot run Docker builds, use Kaniko / BuildKit or a Docker-enabled agent.
- Keep credentials in the Jenkins Credentials store. Do not hardcode secrets in the pipeline.
- Adjust linting policy to fail or warn depending on your team's preference.

Example: run only tests locally

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install pytest
pytest -q
```

For details, see the `Jenkinsfile` in the repository root.

Troubleshooting common pipeline failures

- Docker Desktop paused: Jenkins running on a macOS agent or a macOS-local runner may show failures like "Docker Desktop is manually paused". Unpause Docker Desktop from the Whale menu (or start the Docker daemon) and re-run the job.
- Missing `REGISTRY` variable: If you see errors about `No such property: REGISTRY`, set the `REGISTRY` environment variable in the job (or leave it empty and set `PUSH_IMAGES=false` to skip the push stage). The pipeline also checks for `REGISTRY` before pushing images.

