# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""Shared Maven publish helpers.

Imported by publish_maven_snapshot.py (snapshot path) and MavenUpleveler inside
qnn_ep_uplevel.py (release path).  This module is deliberately stdlib-only so
that publish_maven_snapshot.py can run without a virtualenv.
"""

from __future__ import annotations

import contextlib
import logging
import os
import secrets
import shutil
import stat
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path

__all__ = [
    "generate_checksums",
    "generate_dummy_jars",
    "mvn_deploy_file",
    "qualcomm_ssl_opts",
    "render_settings_xml",
    "rewrite_pom_version",
    "sign_file_gpg",
    "upload_to_maven_central",
]

logger = logging.getLogger(__name__)

_MAVEN_NS = "http://maven.apache.org/POM/4.0.0"

_SETTINGS_TEMPLATE_PATH = Path(__file__).parent / "settings.xml.template"
_JARPOM_TEMPLATE_PATH = Path(__file__).parent / "jarpom.xml"
_CHECKSUMPOM_TEMPLATE_PATH = Path(__file__).parent / "checksumpom.xml"
_ARTIFACTORY_CA_PATH = Path(__file__).parent.parent / "certs" / "artifactory-ca.pem"


def qualcomm_ssl_opts() -> tuple[list[str], Path | None]:
    """Create JVM SSL flags that include the Qualcomm Artifactory CA cert.

    Creates a PKCS12 truststore containing the Qualcomm CA.
    Returns:
        (["-Djavax.net.ssl.trustStore=...",
          "-Djavax.net.ssl.trustStoreType=PKCS12",
          "-Djavax.net.ssl.trustStorePassword=..."],
         truststore_path)

    The truststore_path must be cleaned up by the caller after Maven completes.
    These JVM-level flags are respected by Maven Resolver (unlike Wagon-only
    ssl.insecure flags which have no effect on the resolver transport).

    If truststore creation fails, returns ([], None) with a warning.
    """
    if not _ARTIFACTORY_CA_PATH.exists():
        logger.warning("Qualcomm CA cert not found at %s; SSL may fail", _ARTIFACTORY_CA_PATH)
        return [], None

    # Resolve keytool: prefer PATH, fall back to JAVA_HOME/bin
    keytool = shutil.which("keytool")
    if not keytool:
        java_home = os.environ.get("JAVA_HOME", "")
        if java_home:
            candidate = Path(java_home) / "bin" / "keytool"
            if candidate.exists():
                keytool = str(candidate)
    if not keytool:
        logger.warning("keytool not found in PATH or JAVA_HOME; SSL truststore cannot be created, Maven may fail")
        return [], None

    storepass = secrets.token_hex(16)
    truststore = None

    try:
        # Build PKCS12 truststore with Qualcomm CA
        fd, truststore_str = tempfile.mkstemp(suffix=".p12")
        truststore = Path(truststore_str)
        os.close(fd)
        # Delete the empty file — keytool will create a fresh PKCS12 keystore
        truststore.unlink()

        # Import Qualcomm CA directly into a new PKCS12 truststore.
        # keytool creates the file when it doesn't exist.
        import_result = subprocess.run(
            [
                keytool,
                "-import",
                "-noprompt",
                "-trustcacerts",
                "-alias",
                "qualcomm-root-ca",
                "-file",
                str(_ARTIFACTORY_CA_PATH),
                "-storetype",
                "PKCS12",
                "-keystore",
                str(truststore),
                "-storepass",
                storepass,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        logger.debug("keytool import stdout: %s", import_result.stdout)
        logger.debug("keytool import stderr: %s", import_result.stderr)

        if import_result.returncode == 0 or "already exists" in import_result.stderr:
            # Also import the system CA bundle so Maven Central is reachable.
            # Iterate individual certs from the system bundle since keytool
            # requires one cert per import invocation when using -file.
            java_home = os.environ.get("JAVA_HOME", "")
            system_cacerts = None
            if java_home:
                candidate = Path(java_home) / "lib" / "security" / "cacerts"
                if candidate.exists():
                    system_cacerts = candidate
            if system_cacerts:
                subprocess.run(
                    [
                        keytool,
                        "-importkeystore",
                        "-srckeystore",
                        str(system_cacerts),
                        "-srcstorepass",
                        "changeit",
                        "-destkeystore",
                        str(truststore),
                        "-deststorepass",
                        storepass,
                        "-noprompt",
                    ],
                    check=False,
                    capture_output=True,
                    timeout=30,
                )

        if import_result.returncode == 0 or "already exists" in import_result.stderr:
            logger.info("Built PKCS12 truststore with Qualcomm CA cert at %s", truststore)
            return [
                f"-Djavax.net.ssl.trustStore={truststore}",
                "-Djavax.net.ssl.trustStoreType=PKCS12",
                f"-Djavax.net.ssl.trustStorePassword={storepass}",
            ], truststore
        else:
            logger.debug("keytool import failed (returncode=%d): %s", import_result.returncode, import_result.stderr)

        logger.warning("Could not set up SSL truststore for Qualcomm CA; Maven may fail with certificate errors")
        return [], None

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        logger.warning("Error setting up SSL truststore (%s); Maven may fail with certificate errors", exc)
        if truststore:
            with contextlib.suppress(FileNotFoundError):
                truststore.unlink()
        return [], None


def rewrite_pom_version(pom_path: Path, new_version: str) -> None:
    """Rewrite the <version> element in a Maven POM file.

    Edits the file in-place.  The POM namespace is preserved.  Does not strip
    comments that appear outside the root element, but ElementTree will drop
    inline XML comments — acceptable since we only care about the <version> tag.
    """
    ET.register_namespace("", _MAVEN_NS)
    tree = ET.parse(pom_path)
    root = tree.getroot()
    ns = {"m": _MAVEN_NS}

    version_el = root.find("m:version", ns)
    if version_el is None:
        raise ValueError(f"<version> element not found in {pom_path}")

    old_version = version_el.text
    version_el.text = new_version
    tree.write(pom_path, xml_declaration=True, encoding="UTF-8")
    logger.info("Rewrote POM version: %s -> %s in %s", old_version, new_version, pom_path)


def _fill_template(template_path: Path, replacements: dict[str, str]) -> str:
    """Read a file template and substitute {{key}} placeholders."""
    text = template_path.read_text(encoding="utf-8")
    for key, value in replacements.items():
        text = text.replace("{{" + key + "}}", value)
    return text


@contextlib.contextmanager
def _secure_tempfile(suffix: str = "") -> Iterator[Path]:
    """Yield a 600-mode temp file path; delete it on context exit."""
    fd, path_str = tempfile.mkstemp(suffix=suffix)
    path = Path(path_str)
    try:
        os.close(fd)
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        yield path
    finally:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()


@contextlib.contextmanager
def render_settings_xml(
    username: str,
    password: str,
    repository_url: str,
) -> Iterator[Path]:
    """Render settings.xml.template into a 600-mode temp file.

    Yields the path to the rendered file.  The file is deleted on context exit
    even if an exception is raised — password never persists on disk beyond the
    deploy step.
    """
    content = _fill_template(
        _SETTINGS_TEMPLATE_PATH,
        {"username": username, "password": password, "repository_url": repository_url},
    )
    with _secure_tempfile(suffix="-settings.xml") as settings_path:
        settings_path.write_text(content, encoding="utf-8")
        yield settings_path


def generate_dummy_jars(
    work_dir: Path,
    group_id: str,
    artifact_id: str,
    version: str,
    dry_run: bool = False,
) -> tuple[Path, Path]:
    """Generate dummy -sources.jar and -javadoc.jar for Maven Central compliance.

    Maven Central requires sources and javadoc JARs.  Since this is a prebuilt
    AAR with no Java source to ship, we create dummy JARs containing only a
    README placeholder using the maven-jar-plugin via jarpom.xml.

    Returns (sources_jar_path, javadoc_jar_path).
    """
    jarpom_content = _fill_template(
        _JARPOM_TEMPLATE_PATH,
        {"groupId": group_id, "artifactId": artifact_id, "version": version},
    )
    jarpom_path = work_dir / "jarpom.xml"
    jarpom_path.write_text(jarpom_content, encoding="utf-8")

    # maven-jar-plugin needs a README file in the project base dir
    readme_path = work_dir / "README"
    if not readme_path.exists():
        readme_path.write_text(
            f"{artifact_id} {version} — prebuilt AAR; no Java source available.\n",
            encoding="utf-8",
        )

    if dry_run:
        logger.info("[dry-run] Would run: mvn -f %s package", jarpom_path)
        sources_jar = work_dir / "target" / f"{artifact_id}-{version}-sources.jar"
        javadoc_jar = work_dir / "target" / f"{artifact_id}-{version}-javadoc.jar"
        return sources_jar, javadoc_jar

    _run(["mvn", f"-Dmaven.repo.local={work_dir / '.m2'}", "-f", str(jarpom_path), "package"], cwd=work_dir)

    sources_jar = work_dir / "target" / f"{artifact_id}-{version}-sources.jar"
    javadoc_jar = work_dir / "target" / f"{artifact_id}-{version}-javadoc.jar"

    if not sources_jar.exists():
        raise FileNotFoundError(f"Expected sources jar not found: {sources_jar}")
    if not javadoc_jar.exists():
        raise FileNotFoundError(f"Expected javadoc jar not found: {javadoc_jar}")

    logger.info("Generated dummy jars: %s, %s", sources_jar, javadoc_jar)
    return sources_jar, javadoc_jar


def mvn_deploy_file(
    aar: Path,
    pom: Path,
    sources_jar: Path,
    javadoc_jar: Path,
    group_id: str,
    artifact_id: str,
    version: str,
    repository_id: str,
    repository_url: str,
    settings_xml: Path,
    dry_run: bool = False,
    maven_local_repo: Path | None = None,
    ssl_opts: list[str] | None = None,
) -> None:
    """Run `mvn deploy:deploy-file` to publish to an Artifactory Maven repo.

    Run mvn deploy:deploy-file with an explicit -DpomFile; the POM is supplied by the caller
    (CI: produced by the Android Gradle maven-publish plugin;
    uplevel: copied from the source artifactory repository).

    Credentials are passed via settings_xml (a 600-mode file rendered at runtime
    by render_settings_xml()).  No secret appears in argv.
    """
    if maven_local_repo is not None and not dry_run:
        plugin_cache = maven_local_repo / "org" / "apache" / "maven" / "plugins"
        if plugin_cache.exists():
            logger.info("Clearing Maven plugin cache to remove stale metadata")
            shutil.rmtree(plugin_cache, ignore_errors=True)

    cmd = [
        "mvn",
        "-U",
        "deploy:deploy-file",
    ]

    cmd.extend(
        [
            f"-DgroupId={group_id}",
            f"-DartifactId={artifact_id}",
            f"-Dversion={version}",
            "-Dpackaging=aar",
            f"-Dfile={aar}",
            f"-DpomFile={pom}",
            f"-Dsources={sources_jar}",
            f"-Djavadoc={javadoc_jar}",
            f"-DrepositoryId={repository_id}",
            f"-Durl={repository_url}",
            "-DgeneratePom=false",
            "-s",
            str(settings_xml),
        ]
    )

    if maven_local_repo is not None:
        cmd.insert(1, f"-Dmaven.repo.local={maven_local_repo}")

    if dry_run:
        logger.info("[dry-run] Would run: %s", _redact_settings(cmd))
        return

    # Pass SSL truststore opts via MAVEN_OPTS so they don't appear in argv
    # (and thus are not visible in `ps aux` output).
    env = os.environ.copy()
    if ssl_opts:
        existing = env.get("MAVEN_OPTS", "")
        env["MAVEN_OPTS"] = " ".join([existing, *ssl_opts]).strip()

    _run(cmd, env=env)


def generate_checksums(
    files_dir: Path,
    group_id: str,
    artifact_id: str,
    version: str,
    dry_run: bool = False,
) -> None:
    """Generate .md5 and .sha1 files for every artifact in files_dir.

    Uses checksumpom.xml with checksum-maven-plugin.  The generated checksums
    are written alongside the source files (Maven convention required by the
    Central Portal validator).
    """
    checksumpom_content = _fill_template(
        _CHECKSUMPOM_TEMPLATE_PATH,
        {"groupId": group_id, "artifactId": artifact_id, "version": version},
    )
    checksumpom_path = files_dir.parent / "checksumpom.xml"
    checksumpom_path.write_text(checksumpom_content, encoding="utf-8")

    if dry_run:
        logger.info("[dry-run] Would run: mvn -f %s package", checksumpom_path)
        return

    _run(
        ["mvn", f"-Dmaven.repo.local={files_dir.parent / '.m2'}", "-f", str(checksumpom_path), "package"],
        cwd=files_dir.parent,
    )


def sign_file_gpg(
    file_path: Path,
    gpg_private_key: str,
    gpg_passphrase: str,
    dry_run: bool = False,
) -> Path:
    """Sign a single file with GPG and return the .asc signature path.

    The passphrase is fed via stdin (--passphrase-fd 0) so it never appears
    in the process argv.  The private key is imported via stdin pipe as well.
    """
    sig_path = file_path.with_suffix(file_path.suffix + ".asc")

    if dry_run:
        logger.info("[dry-run] Would GPG-sign: %s -> %s", file_path, sig_path)
        return sig_path

    # Import the key via stdin — no argv exposure
    import_proc = subprocess.run(
        ["gpg", "--batch", "--import"],
        input=gpg_private_key.encode(),
        check=True,
        capture_output=True,
    )
    logger.debug("gpg --import stderr: %s", import_proc.stderr.decode(errors="replace"))

    # Sign via stdin-piped passphrase
    sign_proc = subprocess.run(
        [
            "gpg",
            "--batch",
            "--pinentry-mode",
            "loopback",
            "--passphrase-fd",
            "0",
            "--armor",
            "--detach-sign",
            "--output",
            str(sig_path),
            str(file_path),
        ],
        input=gpg_passphrase.encode(),
        check=True,
        capture_output=True,
    )
    logger.debug("gpg sign stderr: %s", sign_proc.stderr.decode(errors="replace"))
    logger.info("Signed: %s -> %s", file_path, sig_path)
    return sig_path


def upload_to_maven_central(
    bundle_zip: Path,
    bearer_token: str,
    dry_run: bool = False,
) -> str:
    """Upload a signed bundle zip to the Maven Central Portal.

    Uses curl -K <config_file> so the bearer token never appears in argv.
    The config file is a 600-mode temp file that is deleted after the upload.

    Returns the deployment ID string echoed by the Central Portal on success.
    """
    upload_url = "https://central.sonatype.com/api/v1/publisher/upload?publishingType=USER_MANAGED"

    if dry_run:
        logger.info(
            "[dry-run] Would POST %s to %s with bearer token (not shown)",
            bundle_zip,
            upload_url,
        )
        return "dry-run-deployment-id"

    with _secure_tempfile(suffix="-curl-auth.cfg") as cfg_path:
        cfg_path.write_text(
            f'header = "Authorization: Bearer {bearer_token}"\n',
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                "curl",
                "--fail-with-body",
                "-K",
                str(cfg_path),
                "-X",
                "POST",
                upload_url,
                "-H",
                "accept: text/plain",
                "-H",
                "Content-Type: multipart/form-data",
                "-F",
                f"bundle=@{bundle_zip}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    deployment_id = result.stdout.strip()
    logger.info("Maven Central upload succeeded. Deployment ID: %s", deployment_id)
    logger.info("Review staged bundle at: https://central.sonatype.com/publishing/deployments")
    return deployment_id


def _run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> subprocess.CompletedProcess:
    logger.debug("Running: %s (cwd=%s)", _redact_settings(cmd), cwd)
    return subprocess.run(cmd, check=True, cwd=cwd, env=env)


def _redact_settings(cmd: list[str]) -> list[str]:
    """Return a copy of the command with settings.xml paths replaced by <settings.xml>."""
    result = []
    skip_next = False
    for token in cmd:
        if skip_next:
            result.append("<settings.xml>")
            skip_next = False
        elif token == "-s":
            result.append(token)
            skip_next = True
        else:
            result.append(token)
    return result
