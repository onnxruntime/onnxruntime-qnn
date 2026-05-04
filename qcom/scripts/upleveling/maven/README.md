# Maven Release Infrastructure

This directory contains scripts and templates used to publish the
`com.qualcomm.qti:onnxruntime-android-qnn` Maven artifact to:

- **Internal Artifactory** (`https://artifactory-qdc-global.qualcomm.com/artifactory/aisw-maven-virtual/`)
  — every RC (snapshot) and final release.
- **Maven Central** (`https://central.sonatype.com/`) — final releases only.

## Directory contents

| File | Purpose |
|---|---|
| `jarpom.xml` | Generates dummy `-sources.jar` and `-javadoc.jar` (required by Maven Central; AAR is prebuilt) |
| `settings.xml.template` | Maven server credentials template — filled at runtime, never committed |
| `checksumpom.xml` | Generates `.md5` / `.sha1` checksums required by Maven Central |
| `maven_publish_utils.py` | Shared Python helpers (POM rewrite, dummy-jar generation, `mvn deploy:deploy-file`) |
| `publish_maven_snapshot.py` | Snapshot publisher — called from `upload-artifactory-from-github.yml` |
| `pom.xml` | Artifact POM template with `maven-gpg-plugin` for Maven Central signing — rendered at runtime with `{{groupId}}`, `{{artifactId}}`, `{{version}}` |
| `validate_apk.sh` | Validates the published AAR by building the test APK against it |
| `build_apk_from_aar.sh` | Local dev tool: builds both test APKs against a provided local `.aar` file without going through Maven publish |

## GPG signing key — one-time setup

Maven Central requires every artifact to carry a PGP/GPG signature.  The key
must be generated **once** by a release engineer on a trusted workstation — CI
cannot generate keys.

### 1. Generate the key

Follow the official Sonatype guide: <https://central.sonatype.org/publish/requirements/gpg>

```bash
gpg --gen-key
# Recommended settings:
#   Key type:   RSA (default)
#   Expiration: 2y  (matches Sonatype's documented default)
#   Name:       ONNX Runtime QNN Release
#   Email:      <team distribution list>
#   Passphrase: strong random passphrase — store it in a password manager
```

### 2. Remove signing subkeys (important!)

GPG may create a signing subkey, but Maven Central can only verify against the
**primary key**.  After generation, check for and remove signing subkeys:

```bash
gpg --list-secret-keys --keyid-format=long    # look for "ssb" lines with [S] flag
gpg --edit-key <KEY_ID>
  > key 1      # select the signing subkey
  > delkey     # or: revkey
  > save
```

### 3. Publish the public key

Sonatype's supported keyservers (all three must receive the key):

```bash
gpg --keyserver keyserver.ubuntu.com --send-keys <KEY_ID>
gpg --keyserver keys.openpgp.org     --send-keys <KEY_ID>
gpg --keyserver pgp.mit.edu          --send-keys <KEY_ID>
```

### 4. Export for CI

```bash
gpg --armor --export-secret-keys <KEY_ID> > qnn-maven-signing-key.asc

# Add contents to GitHub repo secret:
#   MAVEN_CENTRAL_GPG_PRIVATE_KEY = <contents of qnn-maven-signing-key.asc>
#   MAVEN_CENTRAL_GPG_PASSPHRASE  = <passphrase chosen in step 1>

# Destroy the local copy immediately after uploading to GitHub Secrets
shred -u qnn-maven-signing-key.asc
```

### 5. Key renewal

GPG keys with a 2-year expiry must be renewed before they expire.  Set a
calendar reminder at ~22 months.

```bash
gpg --edit-key <KEY_ID>
  > expire      # pick a new date, e.g. 2y
  > save

# Re-publish to the three keyservers (same commands as step 3)
# Re-export and update the MAVEN_CENTRAL_GPG_PRIVATE_KEY GitHub secret
gpg --armor --export-secret-keys <KEY_ID> > qnn-maven-signing-key.asc
shred -u qnn-maven-signing-key.asc
```

## Required GitHub repository secrets

| Secret | Used for | Where |
|---|---|---|
| `AISW_MAVEN_ARTIFACTORY_USERNAME` | Push to `artifactory-maven-virtual` | snapshot + release |
| `AISW_MAVEN_ARTIFACTORY_PASSWORD` | Push to `artifactory-maven-virtual` | snapshot + release |
| `MAVEN_CENTRAL_BEARER_TOKEN` | Upload bundle to Central Portal | release only |
| `MAVEN_CENTRAL_GPG_PRIVATE_KEY` | Sign artifacts | release only |
| `MAVEN_CENTRAL_GPG_PASSPHRASE` | Sign artifacts | release only |
| `MAVEN_CENTRAL_GPG_KEY_ID` | Optional — aids debug logging | release only |

`MAVEN_CENTRAL_BEARER_TOKEN` is the base64-encoded `userId:encryptedPass` pair
from <https://central.sonatype.com>.  Obtain the encrypted password from your
Central Portal account settings.

## Security notes

- Secrets are **never placed in process `argv`**.  Maven credentials flow via a
  600-mode `settings.xml` file that is deleted after use.  GPG passphrase is
  piped via stdin (`--passphrase-fd 0`).  The Maven Central bearer token flows
  via a 600-mode curl config file (`curl -K`).
- `mvn -X` (debug mode) is deliberately not used in CI — it dumps effective
  settings including decrypted passwords.

## CI runner prerequisites

The self-hosted build runners must satisfy:

1. **Qualcomm Root CA in the Java truststore** — `artifactory-qdc-global.qualcomm.com`
   uses the Qualcomm Root CA G2 (same CA as `re-artifactory.qualcomm.com`).
   If `mvn deploy:deploy-file` fails with an SSL handshake error, import the CA
   cert (`qcom/scripts/upleveling/certs/artifactory-ca.pem`) into the JDK's
   `cacerts` keystore once on the runner:
   ```bash
   keytool -import -noprompt -trustcacerts \
     -alias qualcomm-ca \
     -file qcom/scripts/upleveling/certs/artifactory-ca.pem \
     -keystore "$(dirname $(which java))/../lib/security/cacerts" \
     -storepass changeit
   ```
2. **`gpg` and `zip` present on the runner** — used by the final-release path.
   If missing, install on the runner host (not via `apt` in CI steps).
