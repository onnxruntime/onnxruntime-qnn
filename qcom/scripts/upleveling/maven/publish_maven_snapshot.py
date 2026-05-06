#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""Publish a just-built AAR+POM to the internal Artifactory Maven repository.

This script is the entry point for the snapshot publish path in
qualcomm-internal-upload-artifactory-from-github.yml.  It is intentionally
separate from qnn_ep_uplevel.py — snapshot publish is a one-shot fan-out from
freshly-built CI artifacts and does not fit the "pull version X from server A,
push version Y to server B" shape of the upleveler framework.

Usage (from CI):
    python publish_maven_snapshot.py \\
        --aar-path      build/android-aarch64/Release/.../onnxruntime-android-qnn.aar \\
        --pom-path      build/android-aarch64/Release/.../onnxruntime-android-qnn.pom \\
        --version       2.3.0-SNAPSHOT \\
        --group-id      com.qualcomm.qti \\
        --artifact-id   onnxruntime-android-qnn \\
        --repository-id snapshots \\
        --repository-url https://artifactory-qdc-global.qualcomm.com/artifactory/aisw-maven-virtual/

Credentials are read from environment variables (never passed on the command line):
    AISW_MAVEN_ARTIFACTORY_USERNAME
    AISW_MAVEN_ARTIFACTORY_PASSWORD
"""

import argparse
import contextlib
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

from maven_publish_utils import (
    generate_dummy_jars,
    mvn_deploy_file,
    qualcomm_ssl_opts,
    render_settings_xml,
    rewrite_pom_version,
)


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        level=level,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish an AAR artifact to the internal Artifactory Maven repository."
    )
    parser.add_argument("--aar-path", required=True, type=Path, help="Path to the .aar file")
    parser.add_argument("--pom-path", required=True, type=Path, help="Path to the .pom file")
    parser.add_argument(
        "--version",
        required=True,
        help="Maven version to publish (e.g. 2.3.0-SNAPSHOT)",
    )
    parser.add_argument(
        "--group-id",
        default="com.qualcomm.qti",
        help="Maven group ID (default: com.qualcomm.qti)",
    )
    parser.add_argument(
        "--artifact-id",
        default="onnxruntime-android-qnn",
        help="Maven artifact ID (default: onnxruntime-android-qnn)",
    )
    parser.add_argument(
        "--repository-id",
        default="snapshots",
        choices=["snapshots", "releases"],
        help="Maven repository ID (default: snapshots)",
    )
    parser.add_argument(
        "--repository-url",
        default="https://artifactory-qdc-global.qualcomm.com/artifactory/aisw-maven-virtual/",
        help="Artifactory repository URL",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    username = os.environ.get("AISW_MAVEN_ARTIFACTORY_USERNAME", "")
    password = os.environ.get("AISW_MAVEN_ARTIFACTORY_PASSWORD", "")

    if not args.dry_run:
        if not username:
            logger.error("AISW_MAVEN_ARTIFACTORY_USERNAME is not set")
            sys.exit(1)
        if not password:
            logger.error("AISW_MAVEN_ARTIFACTORY_PASSWORD is not set")
            sys.exit(1)

    if not args.aar_path.exists():
        logger.error("AAR not found: %s", args.aar_path)
        sys.exit(1)
    if not args.pom_path.exists():
        logger.error("POM not found: %s", args.pom_path)
        sys.exit(1)

    with tempfile.TemporaryDirectory(prefix="maven-publish-") as work_str:
        work_dir = Path(work_str)

        # Copy input files into work_dir so we can modify the POM safely
        aar = work_dir / f"{args.artifact_id}-{args.version}.aar"
        pom = work_dir / f"{args.artifact_id}-{args.version}.pom"
        shutil.copy2(args.aar_path, aar)
        shutil.copy2(args.pom_path, pom)

        # Rewrite the POM version to match what we're publishing
        logger.info("Rewriting POM version -> %s", args.version)
        rewrite_pom_version(pom, args.version)

        # Generate dummy sources + javadoc jars (required by Maven Central and
        # good practice for Artifactory too)
        logger.info("Generating dummy -sources.jar and -javadoc.jar")
        sources_jar, javadoc_jar = generate_dummy_jars(
            work_dir,
            args.group_id,
            args.artifact_id,
            args.version,
            dry_run=args.dry_run,
        )

        # Deploy to Artifactory — credentials flow via settings.xml, not argv
        logger.info(
            "Deploying %s:%s:%s to %s",
            args.group_id,
            args.artifact_id,
            args.version,
            args.repository_url,
        )
        if args.dry_run:
            logger.info("[dry-run] Skipping Maven deploy.")
        else:
            ssl_opts, truststore = qualcomm_ssl_opts()
            try:
                with render_settings_xml(username, password, args.repository_url) as settings_xml:
                    mvn_deploy_file(
                        aar=aar,
                        pom=pom,
                        sources_jar=sources_jar,
                        javadoc_jar=javadoc_jar,
                        group_id=args.group_id,
                        artifact_id=args.artifact_id,
                        version=args.version,
                        repository_id=args.repository_id,
                        repository_url=args.repository_url,
                        settings_xml=settings_xml,
                        dry_run=False,
                        maven_local_repo=work_dir / ".m2",
                        ssl_opts=ssl_opts,
                    )
            finally:
                if truststore:
                    with contextlib.suppress(FileNotFoundError):
                        truststore.unlink()

    logger.info("Snapshot publish complete.")


if __name__ == "__main__":
    main()
