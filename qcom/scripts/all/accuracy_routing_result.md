# Accuracy Routing Result, Version 1

The Linux internal-symbol Snapshot lane exports `accuracy_routing.json` by passing
`--emit-routing-result=<path>` to `run_snapshot_accuracy.sh`. The runner exports
its existing operator-group selection on exit. Exporting does not change its
test filter or exit code. Output preparation and export failures log a warning
and retain the original test execution. Reused local output files are cleared
when preparation succeeds, and new results are replaced atomically.

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
| `decision_granularity` | `operator_group`; the runner selects all Accuracy cases of an affected operator |
| `producer` | Checkout SHA, Actions identity, actual build QAIRT/ORT versions, and downloaded golden profile |
| `golden_manifest` | Versions and profile read from the golden store; missing data is represented by null |
| `versions_match` | Both build versions match the manifest and the manifest supplies a profile |
| `mode` | `compare`, `force_accuracy`, or `generate_goldens` |
| `selection_complete` | Whether the runner completed its group selection |
| `runner_exit_code`, `snapshot_exit_code` | Actual execution outcomes, including failure exits |
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

## Initial Consumer Boundary

This change publishes results only. It supplies no Windows consumer or target
skip policy. A later consumer must validate identity, schema, mode, successful
execution, complete selection, versions, profile, and the registered target
universe, then join evidence to a reviewed target policy. Missing or incompatible
evidence must retain target Accuracy execution.

The current `linux-x86_64-htp-simulator` namespace does not establish a v73
execution profile. `producer.target_htp_arch` is therefore null. Windows ARM64
v73 consumption requires explicit architecture evidence and simulator/device
equivalence before any target skip can be enabled. This exporter never infers
architecture from a profile label.

Target Integration and legacy E2E tests continue under their existing execution
policy. Per-operator legacy disposition is separate from artifact publication.
