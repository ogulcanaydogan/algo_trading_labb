# Security Rotation Guide

This file documents immediate steps to rotate and revoke exposed credentials found in the repository.

1) Immediate: Revoke exposed credentials
  - Binance (Spot & Testnet): go to https://www.binance.com/en/my/settings/api-management and delete the API keys listed. Recreate keys if needed and restrict by IP.
  - Kraken: https://www.kraken.com/u/settings/api - delete affected keys and generate new ones.
  - Telegram Bot: https://core.telegram.org/bots#6-botfather - use BotFather to revoke/regenerate the bot token.
  - Anthropic: revoke API keys in Anthropic console or contact support to rotate keys.
  - Finnhub / CryptoPanic: revoke/regenerate keys via their dashboards.

2) Rotate keys used in external services
  - Replace keys promptly in your deployment secret store (GitHub Actions Secrets, HashiCorp Vault, AWS Secrets Manager, etc.).
  - Do NOT re-add plain secrets to `.env` in source control.

3) Clean history (optional, high risk)
  - If the keys were committed in git history, consider rewriting history with `git filter-repo` or `bfg` and force-pushing to remove secrets from past commits. See: https://help.github.com/en/github/authenticating-to-github/removing-sensitive-data-from-a-repository
  - WARNING: rewriting history affects all clones; coordinate with collaborators.

4) Prevent future leaks
  - Keep `.env` in `.gitignore` (already present). Use `.env.example` for templates.
  - Use the provided `.git/hooks/pre-commit` hook to block accidental commits.
  - Add a CI check that scans for common secret patterns (truffleHog, detect-secrets, gitleaks) on PRs.

5) Audit and Monitoring
  - Check provider dashboards for suspicious activity and revoke any API keys that show unknown usage.
  - Rotate keys and update credentials in your deployment and local machines.

6) Contact support if necessary
  - If you believe keys have been abused, contact the providers immediately and follow their incident response steps.

Example commands for revocation (local helper):
```bash
# Remove local env from index if accidentally staged
git reset -- .env

# Scan repo for likely keys (quick grep)
grep -RIn "\(API_KEY\|API_SECRET\|TOKEN\|ANTHROPIC\|BINANCE\|KRAKEN\|TELEGRAM\)" . || true
```

Document completed rotations below with timestamps.

---
last-rotation: 
rotated-by: 
