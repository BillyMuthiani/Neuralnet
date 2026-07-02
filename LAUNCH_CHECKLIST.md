# Launch Checklist

Pre-launch verification checklist for Kronyx.

## Repository

- [x] README.md updated with badges and hero section
- [x] LICENSE file present
- [x] CONTRIBUTING.md documented
- [x] CODE_OF_CONDUCT.md present
- [x] SECURITY.md documented
- [x] ROADMAP.md present
- [x] CHANGELOG.md present
- [x] .gitignore comprehensive

## CI/CD

- [x] tests.yml runs on push
- [x] publish.yml for PyPI releases
- [x] docs.yml for GitHub Pages

## Documentation

- [x] MkDocs configuration
- [x] api_reference.md
- [x] All examples documented
- [x] Installation guide

## Packaging

- [x] pyproject.toml complete
- [x] Version in single source
- [x] Package builds successfully
- [x] Twine check passes

## Verification Steps

### 1. Run Tests

```bash
pytest
ruff check .
mypy kronyx
```

### 2. Build Package

```bash
python -m build
twine check dist/*
```

### 3. Test Installation

```bash
pip install dist/kronyx-*.whl
python -c "import kronyx; print(kronyx.__version__)"
```

### 4. Create Release

1. Update version in `kronyx/version.py`
2. Commit changes
3. Create git tag
4. Push to GitHub
5. Create GitHub Release

### 5. Publish to PyPI

Workflow auto-publishes on GitHub Release.