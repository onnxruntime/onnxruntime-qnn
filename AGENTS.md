# Repository Guidelines

## Project Structure & Module Organization
- Core runtime sources live in `onnxruntime/` (provider code, session/runtime internals, shared APIs).
- QNN-specific build/test orchestration and helper tooling live in `qcom/`:
  - `qcom/build_and_test.py`: primary entrypoint for build, test, and lint tasks.
  - `qcom/ep_build/`: task graph and build task implementations.
  - `qcom/model_test/` and `qcom/samples/`: model validation and usage samples.
- Tests are mainly under `onnxruntime/test/`, including `onnxruntime/test/providers/qnn/` for QNN provider coverage.
- Docs and contributor references are in `docs/` and `docs/execution_providers/`.

## Build, Test, and Development Commands
- List available tasks: `python qcom/build_and_test.py list`
- Build for host platform: `python qcom/build_and_test.py build`
- Run host tests: `python qcom/build_and_test.py test`
- Run QNN-focused provider tests directly: `./onnxruntime_provider_test --gtest_filter=Qnn*`
- Apply lint/format fixes used by CI: `python qcom/build_and_test.py lint_and_fix`
- Optional dry-run for task planning: `python qcom/build_and_test.py --dry-run build`

## Coding Style & Naming Conventions
- C/C++ style is Google-based with project overrides (`.clang-format`); target max line length is 120.
- Python style follows Black/PEP8 with 120-char line length (`pyproject.toml`).
- Use descriptive, scope-prefixed names when useful (e.g., commit prefixes like `[QNN EP]`, `[ABI]`).
- Run local formatting/lint before opening a PR via `lint_and_fix` or pre-commit.

## Testing Guidelines
- Add or update unit tests for every behavior change; this is a hard requirement in contribution docs.
- Prefer focused test runs while iterating (e.g., `--gtest_filter=Qnn*`), then run broader `test` tasks.
- Python tests generally use `unittest` and are executed with `pytest` where applicable.
- Test names should communicate behavior, e.g., `test_<unit>_<expected_behavior>_when_<condition>`.

## Commit & Pull Request Guidelines
- Commit messages are typically imperative and concise, often with optional tags like `[QNN EP]` and an issue/PR reference `( #123 )` style.
- Keep PRs small and reviewable; separate cosmetic-only changes from functional changes.
- PRs must explain motivation, include test evidence, and resolve review comments.
- For non-trivial changes or new APIs, open/discuss an issue before implementation.

## Security & Configuration Tips
- Report security issues privately to `secure@microsoft.com` (do not open public issues).
- Use environment variables such as `QAIRT_SDK_ROOT` and `ORT_PREBUILT_ROOT` for reproducible local builds.
