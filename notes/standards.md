# Python, Django, and AI/ML Coding Standards

This document outlines the coding standards and best practices for modern software development in Python, with specific guidelines for Django web applications and AI/ML projects. It is organized into three parts: general Python standards, web application standards, and AI/ML project standards.

---

## Part I: General Python Standards

This section applies to **all** Python projects.

### 1. Guiding Principles
- **The Zen of Python (PEP 20)**: All code should strive to embody the principles of simplicity, readability, and explicitness. Run `import this` in a Python interpreter to review them.
- **Clarity and Simplicity**: Prefer clear, readable, and maintainable code over overly clever or complex solutions.

### 2. Formatting, Linting & Tooling
- **PEP 8 Compliance**: Code must adhere to `PEP 8`.
- **Formatting**: Use `black` for auto-formatting and `isort` for import sorting.
- **Linting**: Use `ruff` for high-speed linting and error checking.
- **Static Typing**: Use `mypy` or `pyright`. **Type hints are mandatory** for all public functions and class methods.
- **Pre-commit Hooks**: Use `pre-commit` to automate checks locally and block non-compliant commits.
- **CI Pipeline**: Run checks in the following order: `ruff` -> `black --check` -> `isort --check` -> `mypy` -> `pytest`. Block merges on any failure.

### 3. Naming & Docstrings
- **Naming Conventions**:
    -   `Modules/Packages`: `snake_case`
    -   `Classes`: `PascalCase`
    -   `Functions/Variables`: `snake_case`
    -   `Constants`: `UPPER_SNAKE_CASE`
    -   `Private/Internal`: `_leading_underscore`
- **Docstrings (PEP 257)**: All public modules, classes, and functions must have docstrings. Prefer **Google** or **NumPy** style.

### 4. Project Structure (General Purpose/Library)
A `src` layout is recommended to separate package code from tests and scripts.
```
project/
├── pyproject.toml
├── README.md
├── .pre-commit-config.yaml
└── src/
    └── project_pkg/
        ├── __init__.py
        ├── core/         # Pure domain logic, no I/O
        ├── adapters/     # I/O, integration (files, network, db)
        └── utils.py
└── tests/
    ├── __init__.py
    └── test_core.py
```

### 5. Code & Architectural Patterns
- **Clean Architecture**: Separate code into layers:
    -   **Core/Domain**: Pure business logic.
    -   **Services/Use Cases**: Orchestrates application logic.
    -   **Adapters**: Handles all I/O (database, network).
- **Packaging**: For libraries, define an explicit public API using `__all__` in `__init__.py`.
- **Architecture Decision Records (ADRs)**: Document significant architectural decisions in `docs/adr/`.

### 6. Dependencies & Environments
- **Package Management**: Use `Poetry` or `uv` to lock dependencies for reproducibility.
- **Secrets**: Load secrets from environment variables. Use `.env` files **only** for local development (`.gitignore` it).
- **Containerization**: Use Docker for a consistent and reproducible runtime environment.

### 7. Error Handling & Logging
- **Exceptions**: Raise specific, custom exceptions. Avoid blanket `except Exception`. Wrap external calls in narrow `try/except` blocks and re-raise as custom domain exceptions.
- **Logging**: Use the `logging` library. Configure handlers in one place. Do not use `print()` in library code. For applications, use structured logging (JSON) in production.

### 8. Performance & Concurrency
- **Streaming**: Prefer iterators and generators for streaming data to conserve memory.
- **Profiling**: Profile hot paths with `cProfile` or `py-spy` before optimizing.
- **Asyncio**: Use `asyncio` for high-performance, I/O-bound tasks. Do not mix with blocking I/O calls in the same event loop.

### 9. Versioning & Commits
- **Commits**: Use **Conventional Commits** (`feat:`, `fix:`, etc.) for a clean, machine-readable history.
- **Versioning**: Use **Semantic Versioning (SemVer)** for libraries and packages.

---

## Part II: Django & Web Application Standards

This section builds on Part I with guidelines specific to Django and DRF.

### 1. Project Structure (Django Web App)
```
project/
├── manage.py
├── pyproject.toml
├── README.md
└── src/
    ├── project_name/      # Django project root
    │   ├── __init__.py
    │   ├── settings/
    │   │   ├── base.py
    │   │   ├── local.py
    │   │   └── production.py
    │   ├── urls.py
    │   └── asgi.py
    ├── apps/              # Your project's apps
    │   └── orders/
    │       ├── __init__.py
    │       ├── models.py
    │       ├── services.py
    │       ├── selectors.py
    │       ├── api/
    │       │   ├── serializers.py
    │       │   └── views.py
    │       └── tests/
    └── templates/
    └── static/
```

### 2. Settings & Configuration
-   **Structure**: Split settings into `base.py`, `local.py`, `test.py`, and `prod.py`. Select via `DJANGO_SETTINGS_MODULE`.
-   **Security**: `DEBUG=False` in production. Configure `ALLOWED_HOSTS`, `CSRF_TRUSTED_ORIGINS`, and `SECURE_*` headers.

### 3. Models & Migrations
-   **Logic**: Business logic belongs in `services.py`, not models.
-   **Fields**: Use `UUIDField` for public identifiers. Be explicit with `db_table`, `indexes`, and `max_length`.
-   **Signals**: Avoid heavy signals; if used, they must be small and idempotent.
-   **Migrations**: Review migrations as code. Use a zero-downtime strategy (add -> backfill -> switch -> drop).

### 4. ORM & Queries
-   **Selectors**: Place complex read-only queries in `selectors.py`.
-   **Performance**: Aggressively use `select_related` and `prefetch_related`. Use `transaction.atomic()` for multi-step writes.
-   **Raw SQL**: Avoid raw SQL. If necessary, always parameterize it to prevent injection.

### 5. Views & APIs (DRF)
-   **Structure**: Prefer `ViewSets` and `Routers`. Keep views thin; call services for logic.
-   **Serializers**: Use separate `Input` and `Output` serializers if shapes differ. Validation logic belongs in serializers or dedicated `validators.py`.
-   **Pagination**: Always use pagination (`LimitOffset` or `PageNumber`). Do not return unbounded lists.
-   **Permissions & Throttling**: Be explicit. Define custom permission classes and apply throttling to sensitive endpoints.
-   **URL Design**: Use resource-oriented, plural nouns (e.g., `/api/v1/orders/{uuid}/`).

### 6. Tasks & Integrations
- **Background Tasks**: Use `Celery` for background jobs. Tasks must be idempotent and retryable.
- **External Calls**: Wrap external service calls (HTTP, S3) in adapters with circuit breakers/retries (e.g., `tenacity`).

### 7. Testing
- **Framework**: Use `pytest` with `pytest-django`.
- **Data**: Use `factory_boy` for test data fixtures.
- **Database**: Use `pytest.mark.django_db(transaction=True)` to ensure test isolation.
- **API Tests**: Assert status codes, error shapes, and response schemas.

### 8. Security
- **OWASP Top 10**: Mitigate common web vulnerabilities.
- **Authentication**: Start projects with a custom user model (`AbstractUser`).
- **Brute-force Protection**: Use `django-axes` or a similar tool.
- **Dependency Scans**: Run `pip-audit` or `safety` in CI.

---

## Part III: AI/ML Project Standards

This section provides additional standards for projects involving Machine Learning.

### 1. Project Structure (AI/ML)
This structure separates code from ML-specific assets.
```
project/
├─ pyproject.toml
├─ README.md
├─ dvc.yaml                # DVC pipeline definitions
├─ params.yaml             # Hyperparameters
├─ src/
│  └─ project_pkg/
│     ├─ __init__.py
│     ├─ data/             # Data loading & transformation scripts
│     ├─ models/           # Model architecture definitions
│     ├─ training/         # Training and evaluation scripts
│     └─ inference.py      # Inference wrapper
├─ data/                   # (Tracked by DVC, not Git)
├─ models/                 # (Tracked by DVC, not Git)
├─ notebooks/              # Exploratory notebooks (clear outputs before commit)
├─ experiments/            # Experiment configuration files
└─ tests/
```

### 2. Reproducibility
- **Seed Everything**: Seed all random number generators (`random`, `numpy`, `torch`).
- **Track Experiments**: Use `MLflow` or `W&B` to log hyperparameters, metrics, data versions (via `DVC`), and the git commit hash.
- **Version Models**: Store model artifacts with semantic versioning.

### 3. Data Handling & Provenance
- **Immutability**: Treat raw data as immutable.
- **Data Versioning**: Do not commit data to Git. Use `DVC` to version datasets and models.
- **Privacy**: Sanitize and validate all inputs. Mask or drop PII.

### 4. Model Development & Packaging
- **Modular Training**: Separate data loading, model definition, and training loops.
- **Checkpointing**: Save model weights and optimizer state to allow for resumable training.
- **Model Export**: Use standard formats like `ONNX` or `torch.jit.script` for production.

### 5. MLOps & Monitoring
- **Deployment**: For Django-backed ML, load models once at startup. For heavy traffic, use a dedicated model server (`Triton`, `TorchServe`).
- **Drift Detection**: Implement statistical tests to detect data and concept drift.

---

## Appendices

### A. Code Review Checklist (PR)
- [ ] All CI checks pass (lint, type, test).
- [ ] Follows architectural patterns (logic in services).
- [ ] No secrets included.
- [ ] N+1 queries addressed.
- [ ] For ML tasks, experiment metadata is logged.
- [ ] Migrations are non-blocking and reviewed.

### B. References
- [PEP 8 -- Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [OWASP Top Ten](https://owasp.org/www-project-top-ten/)
- [MLflow](https://mlflow.org/)
- [DVC](https://dvc.org/)
- [Architecture Decision Records](https://adr.github.io/)