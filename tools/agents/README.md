# Claude Code Agents for ONNX Runtime QNN EP

This directory contains a set of specialized [Claude Code](https://docs.claude.com/en/docs/claude-code/overview)
subagents tailored to the ONNX Runtime QNN Execution Provider. Each agent encodes
domain knowledge — architecture, coding conventions, build workflow, test patterns,
PR-review heuristics — so that Claude Code stays on-convention when helping with
common QNN EP tasks.

## What's in this directory

| Agent | Purpose |
|-------|---------|
| [`architect.md`](architect.md) | Deep architectural questions, data flow, partitioning, QDQ, plugin EP model, cross-cutting concerns |
| [`op-builder.md`](op-builder.md) | ONNX → QNN operator translation — add/modify op builders, registration, ONNX↔QNN op map |
| [`fusion.md`](fusion.md) | Multi-op fusions (`IQnnNodeGroup`) — pattern matching, `TryFusion`, Gelu/LayerNorm/LPBQ/etc. |
| [`build.md`](build.md) | Windows ARM64 build workflow, CMake GLOB patterns, linker/include/CMake error diagnosis |
| [`unit-test.md`](unit-test.md) | `RunQnnModelTest`, `TestQDQModelAccuracy`, QDQ u8/u16 coverage, failure triage |
| [`pr-review.md`](pr-review.md) | End-to-end PR review via `WebFetch` — correctness, conventions, tests, build impact |
| [`orchestrate.md`](orchestrate.md) | Multi-step task coordinator that delegates to the other specialists |

## Why these exist

Claude Code's default behavior is generic. QNN EP has strong local conventions
(plugin-EP `Ort::` types, `gsl::span` over `std::vector`, `InlinedVector`/`InlinedHashMap`,
Qualcomm copyright header on new files, Windows ARM64 as the one supported build,
tests run from an artifacts directory rather than the build directory, fusions
registered in `qnn_node_group.cc`, op builders in `op_builder_factory.cc`, etc.).
A fresh model session doesn't know any of this, and catching these violations in
review is expensive. Shipping the guidance as agents lets every contributor get
the same on-rails behavior without re-teaching it each session.

## Installing

Claude Code loads subagents from either of two locations:

- **Per-project:** `<repo-root>/.claude/agents/*.md` — scoped to this repo only
- **Global:** `~/.claude/agents/*.md` — available in every project

### Option A — per-project (recommended)

From the root of your local checkout of this repo:

```bash
mkdir -p .claude/agents
cp tools/agents/*.md .claude/agents/
```

Claude Code will auto-discover the agents the next time it starts in this repo.
`.claude/` is gitignored via local settings by most contributors — these copies
stay on your machine, and you pull updates from `tools/agents/` as the canonical
source.

### Option B — global install

```bash
mkdir -p ~/.claude/agents
cp tools/agents/*.md ~/.claude/agents/
```

Every Claude Code session on your machine (any repo, any directory) will see
these agents. Safe because each agent's `description` scopes it to QNN EP triggers,
but per-project is cleaner.

## Using the agents

Once installed, invoke an agent either **implicitly** (Claude Code routes to the
best-matching agent based on the `description` block) or **explicitly**:

```
> @op-builder add a new op builder for ReduceSum
> @fusion walk me through the LayerNorm fusion trigger flow
> @build python .\qcom\build_and_test.py failed with unresolved external — diagnose
> @unit-test write QDQ u8 and u16 tests for a new Gelu variant
> @pr-review https://github.com/onnxruntime/onnxruntime-qnn/pull/123
> @orchestrate implement a Swish fusion end-to-end — code, tests, build
```

Triggers are listed in each agent's front-matter `description`. The orchestrator
is the safe default for any multi-domain task (e.g. "implement X and write tests");
it decomposes into an ordered plan and delegates to the specialists.

## Placeholders used in the agents

The agents reference two paths abstractly — substitute your own values:

- `<repo-root>` — your local checkout of this repo
  (e.g. `C:\work\onnxruntime-qnn`)
- `<artifacts-dir>` — a workspace directory **outside** the build tree that holds
  `copy_artifacts.ps1` plus the copied test binaries and DLLs
  (e.g. `C:\work\QnnEP-Artifacts`). Tests must run from here, not from
  `build/windows-arm64/Release/`, because the copy script assembles the full
  DLL set (ORT core, QNN EP, QNN HTP backend, test binaries) into one place.

## Editing an agent

Agents are plain Markdown with a YAML front-matter block. To tweak behavior —
adjust a coding-convention check, add a new fusion pattern to the registry, update
a build flag — edit the file here in `tools/agents/` and re-sync to your install
location. Keep changes reviewable: agents are prompts, so a small wording change
can meaningfully shift behavior.

The orchestrator's routing table (`orchestrate.md`) should stay in sync with the
specialist roster — if you add or rename an agent, update the table.
