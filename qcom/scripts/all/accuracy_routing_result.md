# Accuracy Routing Result

The standalone internal-symbol Snapshot runner exports `accuracy_routing.json` by passing
`--emit-routing-result=<path>` to `run_snapshot_accuracy.sh`. The runner exports
its testcase-level selection on exit. Exporting does not change its
test filter or exit code. Output preparation and export failures log a warning
and retain the original test execution. Reused local output files are cleared
when preparation succeeds, and new results are replaced atomically.

## Shared Linux CI and coverage flow

The active Ubuntu internal-symbol caller and instrumented coverage job share
`qnn_test_stages.sh` for provider execution, tier scopes, version resolution,
verdict export, and Accuracy selection:

```text
Aligned golden preparation
  -> Non-Snapshot/Accuracy provider tests (legacy, Component, Integration)
  -> Snapshot / Session Snapshot
  -> Verdict JSON
  -> Testcase-level Accuracy selection
  -> Selected Accuracy execution
```

The ordinary Linux job retains its existing model suites between baseline
validation and the dedicated Snapshot/Accuracy runner. Its baseline filter
excludes both routed tiers, which the dedicated runner owns. Coverage retains
its QNN-only baseline scope and executes every stage using its instrumented
binary, without resetting counters between stages; coverage collection follows
all test phases. The baseline scopes differ intentionally, while both use the
same tier exclusions and the same selector.

`export_accuracy_routing.py` builds evidence and `select_accuracy_from_routing.py`
validates the current invocation's verdict, full registered Accuracy inventory,
versions, source SHA, and required Snapshot pairs before emitting a filter.
Neither job routes from operator groups or copies `runner_action` blindly.
The XML-only observation publisher and coverage-specific gate have been
replaced by this shared flow. Windows jobs and production artifact validation
remain on their existing paths.

CI creates a directory under `RUNNER_TEMP`, named `qnn-routing-<run-id>-<attempt>`.
If that path already exists, preparation warns and disables publication rather
than reusing or deleting it. Normal validation still runs. Only a successful export emits the
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
| `evidence_granularity` | `testcase`; evidence is paired by exact suite variant and testcase name |
| `decision_granularity` | `testcase`; describes local selection, not target eligibility |
| `producer` | Checkout SHA, Actions identity, actual build QAIRT/ORT versions, and downloaded golden profile |
| `golden_manifest` | Versions and profile read from the golden store; missing data is represented by null |
| `versions_match` | Both build versions match the manifest and the manifest supplies a profile |
| `mode` | `compare`, `force_accuracy`, or `generate_goldens` |
| `selection_complete` | Whether the runner completed trusted testcase selection; false on selection failure |
| `runner_exit_code`, `snapshot_exit_code` | Actual execution outcomes; baseline failure also invalidates the published CI result |
| `snapshots` | Registered in-scope Snapshot identities and observed `match`, `drift`, or `untrusted` status |
| `accuracy_decisions` | Accuracy identity, paired Snapshot identities with the same suite variant and testcase name, evidence status, and observed runner action |
| `payload_sha256` | SHA-256 of UTF-8 compact JSON for all other fields, with keys sorted recursively |

`snapshot_test_ids` contains only the paired testcase, not the entire operator
group. OpBuilder and Session Snapshot suites map to the corresponding Accuracy
suite by preserving the variant suffix and exact testcase name. If both tiers
contain that same case, both identities are listed. Cases without an exact
pair are `unmapped`; another case or variant cannot supply their evidence.

`accuracy_decisions[].runner_action` is `run`, `skip`, or `unknown` (selection did not
complete). It describes observed Linux testcase selection. An independently
matched case can skip even when another case in the same operator drifts.
Unpaired Accuracy cases remain in the full scoped run set. It does not grant target-platform eligibility. Evidence status
is independently `match`, `drift`, `untrusted`, or `unmapped`; a Linux skip with
missing or incompatible metadata remains untrusted for consumption.

The digest detects payload corruption. It does not authenticate a producer.
Consumers must obtain artifacts from the expected trusted workflow run.
Both job paths use `write_routing_result` for canonical JSON, digest generation,
staging-file naming, and atomic replacement.

Manifest versions must be nonempty strings, and the profile must be a valid
namespace identifier. Invalid nested values are normalized to null. GTest
`failures`, when present, must be an array of objects with string failure
messages; invalid nested evidence is untrusted. This validation never changes
the independently observed runner action.

## Initial Consumer Boundary

This flow routes local Linux Accuracy tests. It supplies no Windows consumer or target
skip policy. A later consumer must validate identity, required fields and types, mode, successful
execution, complete selection, versions, profile, and the registered target
universe, then join evidence to a reviewed target policy. Missing or incompatible
evidence must retain target Accuracy execution.

The initial contract requires producer and consumer scripts from the same
checkout SHA and artifacts from the expected workflow run and attempt. It does
not support consuming artifacts produced by another script revision. There is
no separate schema-version field or cross-version compatibility policy;
provenance and required-field validation remain mandatory.

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

## Local execution policy

The shared runner starts from all registered Accuracy cases in the requested
operator filter. Compare mode skips a case only when every registered exact
Snapshot pair is a trusted match and the golden versions/profile are valid.
Missing, skipped, malformed, or unmapped evidence retains execution. Selection
failure falls back to the full scoped Accuracy filter. Artifact publication
failure does not change selection. `--force-accuracy` and `--generate-goldens`
always run the full scoped Accuracy universe.

Coverage uses the same exporter and selector as the ordinary runner. Its
`generate_coverage.sh --force-accuracy` option retains Snapshot execution and
runs the full Accuracy universe for coverage baselines. Explicit skip and
single-filter diagnostic modes remain available. Golden drift is a routing
signal; genuine Snapshot failures retain full Accuracy fallback and fail the
job after the coverage report is generated. Operator filters remain scope
selectors.

The current `linux-x86_64-htp-simulator` namespace does not establish a v73
execution profile. `producer.target_htp_arch` is therefore null. Windows ARM64
v73 consumption requires explicit architecture evidence and simulator/device
equivalence before any target skip can be enabled. This exporter never infers
architecture from a profile label.

Target Integration and legacy E2E tests continue under their existing execution
policy. Per-operator legacy disposition is separate from artifact publication.
