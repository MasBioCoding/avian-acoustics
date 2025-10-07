# Avian Acoustics – Contributor Guidance

## Scope
These instructions apply to the entire repository unless a more specific `AGENTS.md` is created deeper in the tree.

## Project Overview
This project contains scripts and configurations for collecting and processing bird audio recordings (primarily via the Xeno-Canto API) and for producing embeddings, clustering, and visualisations. Most code is written in Python 3.11 and relies on scientific-computing libraries plus tools for interactive data exploration.

## Coding Standards
- Use Python 3.11 syntax and typing features. Always include type hints for new functions and significant refactors so automated tooling and contributors can reason about data flow easily.
- Format Python code with `black` (line length 88) and keep imports sorted with `isort`. If a formatting conflict arises, prefer readability and consistency with surrounding code.
- Add doctrings for modules, public classes, and functions. Document command-line interfaces with clear argument descriptions.
- When adding configuration files (e.g., YAML), include inline comments for non-obvious settings so newcomers understand their effect.
- Avoid hard-coded local paths in committed scripts; prefer configuration options or environment variables.

## Testing & Validation
- Before committing, run targeted checks relevant to your change. For Python scripts, prefer `python -m compileall <file>` for syntax validation and add lightweight unit or smoke tests when feasible.
- For scripts that call external services (e.g., Xeno-Canto), mock or document how to run them safely to prevent accidental rate-limit issues. If network access is required for a manual test, describe the steps in the PR instead of running them in automated checks.

## Documentation & PR Expectations
- Update README files or script docstrings whenever you introduce new behaviour or dependencies.
- Pull request descriptions should include: a concise summary of user-facing changes, testing steps (or rationale if tests were not run), and any follow-up work or limitations that future contributors should know.

## Data Sensitivity
The repository may interact with large audio datasets. When demonstrating usage or providing examples, avoid committing real user data. Instead, reference public datasets or provide reproducible instructions to regenerate artefacts locally.
