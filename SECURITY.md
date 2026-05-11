# Security Notes — EnergyGuard Fairness Audit Service

This document describes the trust model and known security limitations of v1.

## Trust Model

The v1 service is designed for use **inside a trusted network** (the EnergyGuard
TEF) and behind an authenticating gateway that the operator provides. The service
itself **does not implement authentication, authorisation, or rate limiting**
(explicitly out of scope per `CLAUDE.md`).

Do **not** expose `POST /api/evaluations` or the MLflow UI (`:9007`) to the public
internet without an authenticating reverse proxy in front.

---

## Known Risks

### 1. Pickle deserialisation in uploaded `.joblib` models — arbitrary code execution

`runner/loaders/model_loaders.py` calls `joblib.load(path)` on user-uploaded
`.joblib` files. joblib uses Python's `pickle` module under the hood, and
unpickling untrusted input is equivalent to executing arbitrary Python code
inside the service process.

**This is inherent to the v1 design** (the sklearn ecosystem standardises on
joblib for model serialisation) and cannot be mitigated without abandoning the
joblib format.

**Mitigations in place:**

- The container runs as a non-root user (`uid 1001`, `fairness:fairness`) so a
  malicious model cannot trivially escalate to container-root.
- Upload size is capped at 500 MB to prevent disk exhaustion.
- Uploaded filenames are sanitised with `Path(...).name` to block path
  traversal out of the temp directory.

**Operator responsibilities:**

- Accept models only from trusted users.
- Run the container with read-only root filesystem and dropped capabilities
  (`--read-only --cap-drop=ALL` or equivalent Kubernetes `securityContext`).
- Place the API behind an authenticating gateway.

### 2. Rendered HTML reports may contain user-controlled strings

The HTML report includes `run_name` (from the YAML config) and group labels
(from CSV cell values) in the rendered output. As of `[commit hash]`, Jinja2
auto-escaping is enabled (`select_autoescape`), so user-controlled strings are
HTML-escaped before insertion. Plotly chart fragments are inserted via the
explicit `| safe` filter — these fragments are server-built and never contain
user input.

If you modify `templates/report.html.j2`, **do not add `| safe` to any field
that originates from user input** (config values, dataset cells, warning
messages).

### 3. MLflow UI exposed without authentication on port 9007

The bundled `docker-compose.yml` starts an MLflow tracking server on `:9007`
with no authentication. The fairness service does not write to MLflow in v1
(planned for v1.1), so the UI is empty by default. **Do not expose port 9007
publicly.**

---

## Reporting Vulnerabilities

This is a research project under Horizon Europe GA 101172705. For
security-relevant issues, please open a private GitHub security advisory or
contact the EnergyGuard project coordinator at NTUA EPU.
