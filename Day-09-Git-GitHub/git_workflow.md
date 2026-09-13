# Day 09 — Git & GitHub

## Basic Workflow

```bash
git status
git add .
git commit -m "Day 09: practice Git workflow"
git push
```

## Useful Commands

```bash
git log --oneline
git branch
git switch -c feature/example
git merge feature/example
git diff
```

## Repository Safety

Never commit:

```text
.env
.venv/
API keys
passwords
private credentials
large generated files
```

Use `.gitignore` to keep these out of the repository.

## Portfolio Practice

Each day of the 100-day journey should ideally produce one meaningful commit containing the work for that day.
