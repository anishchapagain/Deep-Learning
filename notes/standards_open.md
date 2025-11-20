#  Python, Django, and General-purpose AI/ML Coding Standards

## 1. Goals & Scope
This document updates and extends the provided Python & Django coding standard to add **best practices for general-purpose AI/ML projects** (data science, experiments, model training, model packaging and inference). It assumes projects are intended to be collaborative, testable, secure, and reproducible.

It covers:
- Formatting, linting, typing and CI
- Project layout patterns for code + ML artifacts
- Django / DRF patterns (where relevant)
- Data handling, experiments, model versioning and reproducibility
- Packaging and deployment guidance for models and inference endpoints
- Observability, security, privacy, and references

---

## 2. Formatting, Linting & Tooling (must)
- Enforce **PEP 8** as baseline.
- Automatic formatters: **Black** (opinionated). Configure `pyproject.toml`.
- Import sorting: **isort**.
- Linting: **Ruff** (fast) or **Flake8** where Ruff lacks rules.
- Static typing: **mypy** or **pyright** — **type hints mandatory** on public functions/classes and module-level APIs.
- Pre-commit: use **pre-commit** to run black, isort, ruff, mypy checks locally and block bad commits.

Example `.pre-commit-config.yaml` hooks:
```yaml
repos:
- repo: https://github.com/psf/black
  rev: stable
  hooks: [{id: black}]
- repo: https://github.com/pycqa/isort
  rev: stable
  hooks: [{id: isort}]
- repo: https://github.com/charliermarsh/ruff-pre-commit
  rev: stable
  hooks: [{id: ruff}]
```

- CI: run `ruff -> black --check -> isort --check -> mypy -> pytest -> security_scans` in this order.

---

## 3. Naming & Docstrings
- Modules/packages: `snake_case`.
- Classes: `PascalCase`.
- Functions/variables: `snake_case`.
- Constants: `UPPER_SNAKE_CASE`.
- Private/internal names: leading underscore `_internal_helper`.
- Docstrings: follow **PEP 257**; prefer **Google** or **NumPy** style for ML code (examples in docstring should include shapes and dtypes).
- Public modules, classes and functions must have docstrings including:
  - short description
  - arguments (types)
  - return (type, shape)
  - raises/errors

Example for ML function:
```py
def preprocess_batch(batch: np.ndarray) -> np.ndarray:
'    """',
    Normalize and pad a batch.

    Args:
        batch (np.ndarray): shape (B, T, F) float32

    Returns:
        np.ndarray: normalized batch, same shape.
'    """',
```

---

## 4. Project Layout (recommended)
A reproducible layout that separates code, experiments, and artifacts:

```
project/
├─ pyproject.toml
├─ README.md
├─ .pre-commit-config.yaml
├─ src/
│  └─ project_pkg/
│     ├─ __init__.py
│     ├─ api/               # django apps or fastapi endpoints
│     ├─ core/              # pure domain logic (no IO)
│     ├─ services/          # business/services layer
│     ├─ adapters/          # IO: storage, HTTP clients, DB
│     ├─ ml/
│     │  ├─ data/           # data loading & transforms
│     │  ├─ models/         # model definitions
│     │  ├─ training/       # training loops, experiment code
│     │  └─ inference/      # inference wrappers, model loaders
│     └─ utils/
├─ experiments/            # ML experiment configs, Hydra or YAML
├─ data/                   # (gitignored) raw/processed references
├─ models/                 # (gitignored) trained model artifacts
├─ notebooks/              # exploratory notebooks (clear outputs before commit)
├─ tests/
└─ .github/workflows/ci.yml
```

- Keep `src/` layout to avoid import pitfalls.
- Keep **separation of concerns**: `core` (pure), `adapters` (IO), `services` (use cases).

---

## 5. Dependencies & Environment
- Use **Poetry** or **pip-tools** to lock dependencies. For reproducible ML experiments, pin exact versions (`poetry.lock`).
- Use `.python-version` (pyenv) or tool-specific pin.
- Manage secrets through environment variables or a secret manager (Vault, AWS Secrets Manager). Use `.env` only for local development; never commit secrets.
- Use containers (Docker) for training & inference to ensure reproducibility. Provide `Dockerfile` and `docker-compose` for dev workflows when relevant.

---

## 6. Imports & Packaging
- Prefer absolute imports; relative imports allowed for tightly-coupled modules.
- Keep `__all__` explicit for library public API.
- Use namespace packages only when publishing multiple distributions.

---

## 7. Type Hints & Runtime Validation
- Add type hints for all public APIs. Use `typing` and `typing_extensions` where necessary.
- For ML data contracts, validate input shapes and types at module boundaries (simple checks or use Pydantic for stricter validation).
- Use `dataclasses` for config objects; consider `pydantic` or `omegaconf` (Hydra) for experiment config validation.

---

## 8. Logging, Observability & Metrics
- Use Python `logging` (structured logging recommended in prod — JSON).
- Add request / run IDs for traceability.
- For ML runs: log hyperparameters, metrics, artifacts to experiment tracking system (MLflow, Weights & Biases, or a simple DB).
- Emit metrics via Prometheus/OpenTelemetry for API/inference performance and model health (latency, error rate, input drift signals).

---

## 9. Testing & Quality
- Testing tools: **pytest**, **pytest-cov**. Aim for meaningful coverage — prefer branch and critical-path coverage over blind % values.
- Test types:
  - Unit: pure functions, services, model logic (fast).
  - Integration: DB, adapters, training loops with small sample data.
  - End-to-end (smoke): full pipeline on a small dataset or a canned fixture.
- Use factories (`factory_boy`) or fixtures for test data.
- Use small deterministic datasets for ML unit tests; avoid training large models in unit tests. Use mocking for heavy IO or time-consuming external calls.
- Names: `function__scenario__expected()` (double underscores).
- For numerical tests, use tolerances and seeded randomness (`np.testing.assert_allclose(..., rtol=..., atol=...)`).

---

## 10. Reproducibility for ML
- Always seed RNGs: Python `random`, NumPy, PyTorch, TensorFlow, and any library-specific RNGs. Document multi-worker determinism caveats.
- Record full environment: Python & package versions, CUDA/cuDNN versions, and hardware specs.
- Use experiment tracking (MLflow, wandb) or store metadata in artifact store. Track:
  - config file (YAML/JSON)
  - git commit hash
  - data version (DVC hash or storage path)
  - hyperparameters
  - metrics and model artifact checksum
- Save model artifacts in a structured location with semantic versioning (`models/{model_name}/v{semver}/...`).

---

## 11. Data Handling & Provenance
- Treat raw data as **immutable**; write processed outputs to new files with hashes and metadata.
- Don’t commit data to git. Use DVC or similar to manage dataset versions. For small datasets, store checksums in git.
- Log data transformations as code and as experiment metadata. Prefer deterministic transformations and store transformation pipelines (e.g., scikit-learn `Pipeline` or custom `transform` classes).
- Sanitize and validate external inputs; drop PII or mask it according to privacy policy/regulations.

---

## 12. Model Development & Training
- Training code should be modular: data loader, model definition, training loop, evaluation, and checkpointing should be separately testable.
- Use checkpointing: save both model weights and optimizer/scheduler state for resumability.
- Keep training loops deterministic where feasible (seed, controlled data shuffling).
- Use callbacks/hooks for:
  - early stopping
  - model checkpointing
  - logging metrics
- For heavy experiments use cluster/job schedulers (Kubernetes, Slurm) and containerized images.

---

## 13. Model Evaluation & Validation
- Define clear evaluation metrics and baselines before training.
- Split data with deterministic folds and keep validation/test splits immutable.
- Evaluate fairness and bias metrics where applicable.
- Validate generalization (out-of-distribution checks) and monitor training/validation curves.

---

## 14. Model Packaging & Inference
- Export models with a clear contract. Options:
  - PyTorch: `state_dict()` + class code OR `torch.jit.script`/`torchscript` for production
  - TensorFlow: `SavedModel`
  - Scikit-learn: `joblib` or export pipeline via ONNX for language-agnostic serving
- Provide a lightweight inference wrapper that:
  - loads model artifact and config
  - validates input schema
  - runs preprocessing deterministically
  - returns typed outputs with error handling
- Keep inference code separate from training code.
- For latency-sensitive endpoints, consider model optimizations (TorchScript, ONNX Runtime, quantization).

---

## 15. Deployment Patterns
- For Django-backed ML APIs:
  - Keep heavy model loading out of request path (load once at process start or via a model server).
  - Use async workers or background tasks for long-running tasks (Celery/Redis/Queue).
  - Cache model predictions if inputs repeat and it’s safe.
- For larger scale, use a dedicated model server (TorchServe, TensorFlow Serving, Triton) and let Django orchestrate auth, routing, and business logic.

---

## 16. Monitoring & Drift Detection
- Monitor:
  - Latency, error rates, saturation.
  - Model performance metrics (prediction distributions, top-k accuracy if applicable).
  - Data drift and concept drift (statistical tests on input feature distributions).
- Alert on abnormal changes and provide automated rollback/runbook steps.

---

## 17. Security & Privacy
- Do not eval/exec arbitrary data.
- Sanitize inputs and validate sizes; guard against large payloads and DoS.
- Protect PII: follow regulatory compliance and encryption in transit and at rest.
- Use secure defaults: `DEBUG=False` in Django, secure cookies, HSTS, CSRF protections.
- Run dependency vulnerability scans: `pip-audit`, `safety`, or GitHub Dependabot.

---

## 18. Performance & Cost Control
- Profile before optimizing (py-spy, cProfile).
- For training cost control:
  - Use mixed precision where safe.
  - Use efficient data loading and caching.
  - Use batch-sizing strategies and gradient accumulation if memory constrained.

---

## 19. CI/CD, Releases & Versioning
- CI pipeline should:
  - Run linters, type checks, tests and security scans.
  - Build artifacts (wheels, docker images) on tagged commits or main-ready merges.
- Versioning:
  - Use semantic versioning for packages.
  - Use model semantic versions and store model metadata.
- Data migrations and model migrations must be reviewed; strategy: add → backfill → switch → drop (zero-downtime).

---

## 20. Code Review Checklist (PR)
- [ ] Linting and formatting passed (ruff/black/isort).
- [ ] Type checks pass (mypy/pyright).
- [ ] Tests added/updated; CI green.
- [ ] No secrets in PR.
- [ ] Business logic in services, not views/models.
- [ ] Queries optimized; no N+1.
- [ ] Serializers validate inputs (for Django/DRF).
- [ ] Model training code deterministic for tests; long-running jobs isolated.
- [ ] Experiment metadata logged (git commit, config).
- [ ] Migrations reviewed.

---

## 21. Helpful Patterns & Examples
- Keep `services.py` for use-case orchestration; `selectors.py` for read-only query helpers.
- Prefer dependency injection for testability (pass clients/clients wrappers into services).
- For ML pipelines use config-driven runs (Hydra/omegaconf, MLflow projects, or simple YAML).
- Use small canned fixtures and mock I/O in unit tests.

---

## 22. Recommended Tools & Libraries
- Formatting & lint: `black`, `isort`, `ruff`, `pre-commit`
- Typing & checks: `mypy`, `pyright`
- Testing: `pytest`, `pytest-cov`, `hypothesis` (for property tests)
- Django: `django`, `djangorestframework`, `pytest-django`, `factory_boy`
- ML: `numpy`, `pandas`, `scikit-learn`, `PyTorch` / `TensorFlow`, `onnx`
- Experiment & data: `MLflow`, `Weights & Biases`, `DVC` (data version control), `Hydra`/`omegaconf`
- Serving: `TorchServe`, `TensorFlow Serving`, `Triton`, `FastAPI` for microservices
- Observability: `Prometheus`, `OpenTelemetry`, `Grafana`
- Security: `bandit`, `pip-audit`, `safety`

---

## 23. References (authoritative)
- PEP 8 — Style Guide for Python Code: https://peps.python.org/pep-0008/
- PEP 257 — Docstring Conventions: https://peps.python.org/pep-0257/
- Black formatter: https://black.readthedocs.io/
- isort: https://pycqa.github.io/isort/
- Ruff: https://github.com/charliermarsh/ruff
- mypy: https://mypy.readthedocs.io/
- Django docs: https://docs.djangoproject.com/
- Django REST framework: https://www.django-rest-framework.org/
- MLflow: https://mlflow.org/
- DVC: https://dvc.org/
- Hydra (config management): https://hydra.cc/
- OWASP: https://owasp.org/
- ONNX: https://onnx.ai/

---

## 24. Minimal `.gitignore` recommendations
```
__pycache__/
*.pyc
.env
venv/
.poetry/
.pytest_cache/
models/
data/
*.ckpt
*.pt
*.pth
```

---

## 25. Closing notes
- Prefer readability, reproducibility and observability over clever micro-optimizations.
- When in doubt, write a short ADR (Architecture Decision Record) and link it from code comments or the PR.
- Keep experiment artifacts and model metadata discoverable and immutable (by using object storage + metadata DB).