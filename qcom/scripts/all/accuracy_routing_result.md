# Accuracy Routing Result, Version 1

The standalone internal-symbol Snapshot runner exports `accuracy_routing.json` by passing
`--emit-routing-result=<path>` to `run_snapshot_accuracy.sh`. The runner exports
its existing operator-group selection on exit. Exporting does not change its
test filter or exit code. Output preparation and export failures log a warning
and retain the original test execution. Reused local output files are cleared
when preparation succeeds, and new results are replaced atomically.

The active `x86_64_ubuntu_22_04` internal-symbol CI caller uses ordinary provider
CTest validation. The unused Linux-only `x86_64` shadow preparation and two-pass
execution branches have been removed from the single-os workflow; standalone
build/test commands and the separate golden producer remain supported. Its publisher
reads the newly generated `onnxruntime_provider_test.<config>.results.xml` through
`observe_provider_routing.py`; it does not execute Snapshot or Accuracy tests
again. The previous report is removed before validation so an interrupted
invocation cannot publish stale evidence. Missing or malformed reports and
export failures warn without changing the validation verdict.

This producer records `mode: provider_observation`,
`decision_granularity: none`, `selection_complete: false`, and Accuracy
`runner_action: unknown`. It observes Snapshot evidence, not a routing decision.
`snapshot_exit_code` is null because the provider exit is aggregate;
`provider_validation_exit_code` and `runner_exit_code` record the validation
command outcome. It does not enable golden preflight or alter the current
CTest filter. Without explicit golden metadata, Accuracy evidence remains
untrusted. Consumers must retain target execution for this observation mode.
Enabling active testcase routing is separate from publishing its observations.

CI creates a unique directory under `RUNNER_TEMP`, named with the run ID,
attempt, and a random suffix. Only a successful export emits the
`routing_result_path` step output used by the upload action. Preparation or
export failure therefore cannot publish a previous result. An `always()` cleanup
step removes only the current directory after upload; cleanup failure warns.
The runner's normal `RUNNER_TEMP` cleanup provides an additional cleanup path.

The Actions artifact name is
`qnn-routing-linux-x86_64-<config>-py<python-version>-<run-attempt>`. Consumers must
also validate the embedded repository, checkout commit, workflow reference, job,
run ID, and attempt against the expected producer run.

## Contract

| Field | Meaning |
|---|---|
| `schema_version` | `1`; other versions require an explicit consumer implementation |
| `evidence_granularity` | `testcase`; evidence is paired by exact suite variant and testcase name |
| `decision_granularity` | `operator_group` for the legacy two-pass runner, `none` for active provider observation; neither grants target skip authorization |
| `producer` | Checkout SHA, Actions identity, actual build QAIRT/ORT versions, and downloaded golden profile |
| `golden_manifest` | Versions and profile read from the golden store; missing data is represented by null |
| `versions_match` | Both build versions match the manifest and the manifest supplies a profile |
| `mode` | `compare`, `force_accuracy`, `generate_goldens`, or `provider_observation`; observation cannot authorize target skips |
| `selection_complete` | Whether the runner completed its group selection |
| `runner_exit_code`, `snapshot_exit_code` | Actual execution outcomes; Snapshot exit is null when only an aggregate provider verdict is available |
| `snapshots` | Registered in-scope Snapshot identities and observed `match`, `drift`, or `untrusted` status |
| `accuracy_decisions` | Accuracy identity, paired Snapshot identities with the same suite variant and testcase name, evidence status, and observed runner action |
| `payload_sha256` | SHA-256 of UTF-8 compact JSON for all other fields, with keys sorted recursively |

`snapshot_test_ids` contains only the paired testcase, not the entire operator
group. OpBuilder and Session Snapshot suites map to the corresponding Accuracy
suite by preserving the variant suffix and exact testcase name. If both tiers
contain that same case, both identities are listed. Cases without an exact
pair are `unmapped`; another case or variant cannot supply their evidence.

`accuracy_decisions[].runner_action` is `run`, `skip`, `unknown` (selection did not
complete), or `out_of_scope` (no in-scope Snapshot group). It describes observed
Linux group selection. A paired case may match while its group still runs due
to drift in another case. It does not grant target-platform eligibility. Evidence status
is independently `match`, `drift`, `untrusted`, or `unmapped`; a Linux skip with
missing or incompatible metadata remains untrusted for consumption.

The digest detects payload corruption. It does not authenticate a producer.
Consumers must obtain artifacts from the expected trusted workflow run.

Manifest versions must be nonempty strings, and the profile must be a valid
namespace identifier. Invalid nested values are normalized to null. GTest
`failures`, when present, must be an array of objects with string failure
messages; invalid nested evidence is untrusted. This validation never changes
the independently observed runner action.

## Initial Consumer Boundary

This change publishes results only. It supplies no Windows consumer or target
skip policy. A later consumer must validate identity, schema, mode, successful
execution, complete selection, versions, profile, and the registered target
universe, then join evidence to a reviewed target policy. Missing or incompatible
evidence must retain target Accuracy execution.

### Testcase-level target skip rule

A target consumer starts from its full registered Accuracy scope and defaults
every testcase to `run`. It adds a testcase to the skip list only after all
artifact trust checks pass and the reviewed target policy supplies an eligible
mapping whose required Snapshot identities are all present and trusted matches.
The consumer must check the policy's complete required evidence set; a partial
producer pairing cannot establish completeness on its own.

Drift, missing evidence, an unmapped target case, or any validation failure
retains execution. One matched case may be eligible even when another case in
the same operator group drifts. Neither `runner_action=skip` nor
`runner_action=run` determines the target action. In shadow validation, the
consumer reports this proposed skip list while all target Accuracy tests run;
enforcement is a separate change.

This exporter supplies testcase evidence, not a target-authorized skip list.

The current `linux-x86_64-htp-simulator` namespace does not establish a v73
execution profile. `producer.target_htp_arch` is therefore null. Windows ARM64
v73 consumption requires explicit architecture evidence and simulator/device
equivalence before any target skip can be enabled. This exporter never infers
architecture from a profile label.

Target Integration and legacy E2E tests continue under their existing execution
policy. Per-operator legacy disposition is separate from artifact publication.
