# Agent Guidelines

This document contains guidelines and processes for all agents working on the algorithmic trading lab repository.

## Repository Structure

- **bot/**: Core trading engine and strategy implementations
- **api/**: FastAPI backend with REST endpoints and WebSocket support
- **tests/**: Test suite with unit, integration, and property-based tests
- **docs/**: Documentation and architectural decisions
- **scripts/**: Runnable scripts for trading, ML, and utilities
- **infra/**: Infrastructure and deployment configurations

## Development Commands

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=bot --cov=api

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m slow
```

### Code Quality
```bash
# Linting
ruff check bot/ api/ tests/

# Type checking
mypy bot/ api/

# Format code
ruff format bot/ api/ tests/
```

### Environment Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install dev dependencies
pip install -e ".[dev]"
```

## Agent Process

1. Always read AGENTS.md, TASK.md, and HANDOFF.md first
2. Restate objective and acceptance criteria from TASK.md
3. Follow instructions from HANDOFF.md exactly
4. Run standard test/lint commands before making changes
5. Work autonomously until acceptance criteria are met
6. Update TASK.md and HANDOFF.md at session end

## Code Standards

- Use ruff for linting and formatting
- Maintain test coverage above 40%
- Follow existing architectural patterns (see docs/DECISIONS.md)
- Use type hints for all public functions
- Add comprehensive error handling with specific exceptions
- Include correlation IDs in logging for request tracking

## Security Guidelines

- Never commit API keys or secrets
- Validate all user inputs
- Use proper authentication for all endpoints
- Implement rate limiting on critical APIs
- Follow OWASP best practices

## Testing Strategy

- Unit tests for individual components
- Integration tests for trading workflows
- Property-based tests for mathematical invariants
- Performance tests for latency-sensitive operations
- Failure scenario testing for resilience

## Dependencies

- Python 3.10+
- FastAPI for API layer
- SQLite for data persistence
- ccxt for exchange integration
- pytest for testing
- ruff for code quality

## Port Management

- Check port-map.md before configuring new ports
- Document all port conflicts in HANDOFF.md
- Use standard ports: 8000 (API), 5432 (DB), 6379 (Redis)