# Windows ARM64 v73 Accuracy Routing Shadow

The per-PR `Windows ARM64 v73 routing shadow` job observes routing after both the
active Ubuntu internal-symbol producer and Windows ARM64 internal-symbol test lane finish.
The existing build/test jobs run in parallel. The observer reads artifacts and
publishes `observation.json`; it never writes test filters or invokes inference.

The Linux dependency is `build-and-test-linux-x86_64-ubuntu-22-04`, and the
producer retains the `qnn-routing-linux-x86_64` artifact namespace; the
download name includes Release, the caller's Python
version, and run attempt. Main still has the Windows internal-symbol shadow
caller; its full Accuracy execution supplies the target evidence. If #932
removes that caller, this observer must be retargeted to its replacement.

## Target Evidence

`collect_routing_target.py` records the actual Windows build QAIRT/ORT versions
through the shared version resolver, plus checkout and Actions identity. This
small context travels alongside the temporary internal-symbol test archive.

The test runner records observation start time before its normal test task.
Afterward, the collector reads fresh Accuracy JUnit reports across the provider
shards and the actual HTP architecture printed in `LastTest.log`. Missing or
ambiguous architecture, stale reports, or duplicate testcase identities make
the evidence incomplete. No architecture is inferred from runner labels.

## Selection Policy

`v73_accuracy_shadow_policy.json` contains exact target testcase IDs. Its default
is `unresolved`, and its initial `cases` map is empty. Clip is the first audit
candidate; this PR approves no simulator/device equivalence mappings.

A future reviewed case entry has these fields:

```json
{
  "responsibility": "simulator_gated_common",
  "reviewed": true,
  "equivalence_evidence": "reference to reviewed testcase evidence",
  "snapshot_test_ids": ["QnnSnapshot_Example_OpBuilderTest.Case"]
}
```

The alternative responsibilities are `always_run_platform_specific` and
`unresolved`. Both propose running on the target regardless of graph match.

The selector proposes a skip only when producer, target evidence, and target
build match the expected repository, checkout commit, workflow SHA/ref, run ID, and attempt;
the artifact's required fields, payload digest, successful runner outcome, and comparison
mode are valid; versions and an explicit v73 profile agree; and the exact
reviewed mapping connects matched Snapshot evidence to the observed target
Accuracy testcase. A mismatch or missing input proposes run/fallback.

Producer and consumer scripts must come from the same checkout/run; the Linux
artifact has no `schema_version` field. Target evidence and policy retain their
own local format markers. The merged #1141 producer exports
`mode: provider_observation`, `decision_granularity: none`, incomplete selection,
unknown runner actions and unknown target architecture. These are valid
observations but never authorize skips. A routing-capable, architecture-aligned
producer and reviewed equivalence evidence are required before real input can
yield a skip proposal; adaptation to #1156's producer is separate.

## Observation Report

- `would_skip` / `would_run`: proposed selection counts. An unavailable target
  testcase universe is reported as unknown rather than zero tests to run.
- `cases`: exact target IDs, proposed action, actual outcome, and reasons.
- `false_skip_candidates`: skip proposals whose actual target outcome did not
  pass, including skipped or failed cases. These require investigation.
- `fallback_reasons`: input, identity, version, profile, or policy failures.
- `target_execution_changed`: always false in this shadow implementation.

The report is uploaded as
`qnn-routing-windows-arm64-v73-shadow-<run-attempt>`. Download or collection errors
remain visible in this non-required observation lane and retain the existing
full target test execution. Enforced skipping and Clip legacy disposition are
separate follow-up PRs.
