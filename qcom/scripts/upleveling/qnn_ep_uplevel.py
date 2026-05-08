#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

"""
Artifact upleveling script with class-based architecture.
Supports Python wheels, NuGet packages, ZIP archives, TGZ archives, and Maven AAR artifacts.
"""

import argparse
import contextlib
import logging
import os
import re
import shutil
import ssl
import subprocess
import tempfile
import zipfile
from abc import ABC, abstractmethod
from configparser import ConfigParser
from pathlib import Path
from typing import ClassVar
from urllib.parse import urlparse

import requests
from maven import maven_publish_utils
from requests.auth import HTTPBasicAuth

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ARTIFACTORY_CERTS_FILE = os.path.join(SCRIPT_DIR, "certs", "artifactory-ca.pem")
PYPI_RC_FILE = os.path.join(SCRIPT_DIR, ".pypirc")
INI_FILE = os.path.join(SCRIPT_DIR, "config.ini")

# Artifact format mappings
ARTIFACTORY_PREFIXES = {
    "wheel": "re-artifactory-pypi",
    "nuget": "re-artifactory-nuget",
    "zip": "re-artifactory-zip",
    "tgz": "re-artifactory-zip",
}

ARTIFACT_SUFFIXES = {"wheel": ".whl", "nuget": ".nupkg", "zip": ".zip", "tgz": ".tgz"}


class ConfigManager:
    """Manages configuration file reading and URL generation."""

    def __init__(self, pypi_rc_file: str, ini_file: str):
        self.config = ConfigParser()
        self.config.read([pypi_rc_file, ini_file])

    def get_repository_url(self, index: str, product_name: str, version: str) -> str:
        """Get repository URL from configuration."""
        base_url = self.config.get(index, "repository")
        return f"{base_url}/{product_name}/{version}"


class CredentialManager:
    """Manages credentials for different repository types."""

    @staticmethod
    def is_pypi_index(repository_index: str) -> bool:
        """Check if repository is a PyPI index."""
        return repository_index in ["testpypi", "pypi"]

    @staticmethod
    def is_nuget_index(repository_index: str) -> bool:
        """Check if repository is a NuGet index."""
        return repository_index in ["testnuget", "nuget"]

    @staticmethod
    def is_maven_index(repository_index: str) -> bool:
        """Check if repository is a Maven Central index (bearer-token auth, not Artifactory basic auth)."""
        return repository_index in ["maven-central"]

    @staticmethod
    def get_credentials(repository_index: str) -> tuple[str, str]:
        """Get credentials for the specified repository from environment variables."""
        if CredentialManager.is_pypi_index(repository_index):
            # For PyPI repositories, use __token__ as username and API key from environment
            if repository_index == "pypi":
                api_key = os.environ.get("PYPI_API_KEY", "")
            else:  # testpypi
                api_key = os.environ.get("TEST_PYPI_API_KEY", "")
            return "__token__", api_key
        elif CredentialManager.is_nuget_index(repository_index):
            # For NuGet repositories, get API key from environment
            if repository_index == "nuget":
                api_key = os.environ.get("NUGET_API_KEY", "")
            else:  # testnuget
                api_key = os.environ.get("TEST_NUGET_API_KEY", "")
            # For public NuGet repositories, use the API key as the username.
            return api_key, api_key
        elif repository_index == "artifactory-maven-virtual":
            user = os.environ.get("AISW_MAVEN_ARTIFACTORY_USERNAME", "")
            password = os.environ.get("AISW_MAVEN_ARTIFACTORY_PASSWORD", "")
            return user, password
        elif CredentialManager.is_maven_index(repository_index):
            # Maven Central uses a Bearer token, not basic auth.
            # Upload credentials are read directly in MavenUpleveler._upload_maven_central().
            raise ValueError(
                "maven-central does not use basic-auth credentials; use MAVEN_CENTRAL_BEARER_TOKEN directly."
            )
        else:
            # For Artifactory, get username and password from environment
            artifactory_user = os.environ.get("ARTIFACTORY_USERNAME", "")
            artifactory_password = os.environ.get("ARTIFACTORY_PASSWORD", "")
            return artifactory_user, artifactory_password


class ArtifactUpleveler(ABC):
    """
    Base class for artifact upleveling operations.
    Handles common operations like downloading, version updating, and uploading.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config_manager = ConfigManager(PYPI_RC_FILE, INI_FILE)
        self.credential_manager = CredentialManager()

        # Generate URLs
        self.url_from = self.config_manager.get_repository_url(
            args.index_server_from, args.product_name, args.version_from
        )
        self.url_to = self.config_manager.get_repository_url(args.index_server_to, args.product_name, args.version_to)

        # Filter URLs for display
        self.url_from_display = self._filter_url(self.url_from)
        self.url_to_display = self._filter_url(self.url_to)

    def _get_credentials(self, repository_index: str) -> tuple[str, str]:
        """Helper method to get credentials from environment variables."""
        return self.credential_manager.get_credentials(repository_index)

    @property
    @abstractmethod
    def artifact_format(self) -> str:
        """Return the artifact format (wheel, nuget, zip)."""

    @property
    def artifact_suffix(self) -> str:
        """Return the file suffix for this artifact type."""
        return ARTIFACT_SUFFIXES[self.artifact_format]

    @property
    def needs_version_update(self) -> bool:
        """Check if version update is needed."""
        return self.args.version_from != self.args.version_to

    def _filter_url(self, url: str) -> str:
        """Remove API-specific parts from URL for display."""
        return url.replace("api/pypi/", "").replace("api/nuget/", "")

    def _get_ssl_verify(self) -> str:
        """Get SSL verification path based on repository type."""
        if self.credential_manager.is_pypi_index(self.args.index_server_from.lower()):
            return ssl.get_default_verify_paths().cafile
        return ARTIFACTORY_CERTS_FILE

    def download_artifacts(self, url: str, download_dir: str) -> list[str]:
        """Download artifacts from the specified URL."""
        logging.info(f"Downloading {self.artifact_format}s from: {url}")

        auth_credentials = HTTPBasicAuth(*self._get_credentials(self.args.index_server_from))
        verify = self._get_ssl_verify()

        # Fetch artifact list
        response = requests.get(url, auth=auth_credentials, verify=verify)
        if response.status_code != 200:
            raise RuntimeError(f"Unable to fetch artifacts from {url}")

        # Extract artifact file links from Artifactory's HTML directory listing.
        # Parse href values rather than splitting on quotes to handle varied HTML formatting.
        artifact_list = [
            m.group(1)
            for m in (
                re.search(r'href=["\']([^"\']*' + re.escape(self.artifact_suffix) + r')["\']', line)
                for line in response.text.splitlines()
                if self.artifact_suffix in line
            )
            if m
        ]

        if not artifact_list:
            raise RuntimeError(
                f"Expected to find at least one artifact_file for version {self.args.version_from} "
                f" to uplevel, got none"
            )

        # Download each artifact
        for artifact_file in artifact_list:
            url_path = f"{url}/{artifact_file}"
            download_path = os.path.join(download_dir, artifact_file)

            logging.info(f"Downloading {artifact_file}")
            response = requests.get(url_path, auth=auth_credentials, verify=verify)

            if response.status_code != 200:
                raise RuntimeError(f"Unable to fetch {artifact_file} from {url_path}")

            with open(download_path, "wb") as f:
                f.write(response.content)

            if not os.path.exists(download_path):
                raise RuntimeError(f"Failed to download {artifact_file}")

            logging.info(f"Download complete for {artifact_file}")

        return artifact_list

    @abstractmethod
    def update_artifacts(self, artifact_list: list[str], input_dir: str, output_dir: str) -> None:
        """Update artifact versions. Must be implemented by subclasses."""

    @abstractmethod
    def upload_artifacts(self, distribution_dir: str) -> None:
        """Upload artifacts to repository. Must be implemented by subclasses."""

    def run(self) -> None:
        """Execute the complete upleveling workflow."""
        with tempfile.TemporaryDirectory(prefix="run_upleveling_") as tmp_dir:
            # Download artifacts
            artifact_list = self.download_artifacts(self.url_from_display, tmp_dir)

            # Determine upload directory
            upload_dir = tmp_dir
            if self.needs_version_update:
                logging.info(
                    f"Updating {self.artifact_format}(s) version from "
                    f"'{self.args.version_from}' to '{self.args.version_to}'"
                )
                upload_dir = os.path.join(os.path.abspath(os.path.curdir), f"updated_{self.artifact_format}s")
                if os.path.exists(upload_dir):
                    shutil.rmtree(upload_dir)
                os.mkdir(upload_dir)

                # Update artifacts with new version
                self.update_artifacts(artifact_list, tmp_dir, upload_dir)

            # Upload artifacts
            logging.info(f"Uploading {self.artifact_format}s to {self.url_to_display}")
            self.upload_artifacts(upload_dir)

        logging.info(f"Up-leveling for {self.artifact_format} completed successfully!")


class WheelUpleveler(ArtifactUpleveler):
    """Handles PyPI wheel package upleveling."""

    @property
    def artifact_format(self) -> str:
        return "wheel"

    def update_artifacts(self, artifact_list: list[str], input_dir: str, output_dir: str) -> None:
        """Update wheel package versions."""
        for wheel_file in artifact_list:
            logging.info(
                f"Updating version from {self.args.version_from} to {self.args.version_to} for wheel {wheel_file}"
            )

            with tempfile.TemporaryDirectory(prefix="upload_wheel_artifact_") as tmp_dir:
                wheel_path = os.path.join(input_dir, wheel_file)

                # Extract wheel
                with zipfile.ZipFile(wheel_path, "r") as zip_ref:
                    zip_ref.extractall(tmp_dir)

                # Find and update dist-info directory
                dist_info_files = [f for f in os.listdir(tmp_dir) if ".dist-info" in f]
                if len(dist_info_files) != 1 or not dist_info_files[0].endswith(".dist-info"):
                    raise RuntimeError(f"Unable to find dist info for {wheel_file} to update version")

                dist_info_file = dist_info_files[0]
                metadata_file = os.path.join(tmp_dir, dist_info_file, "METADATA")
                record_file = os.path.join(tmp_dir, dist_info_file, "RECORD")

                # Update METADATA
                with open(metadata_file) as f:
                    metadata = f.read()
                with open(metadata_file, "w") as f:
                    f.write(metadata.replace(self.args.version_from, self.args.version_to))

                # Update RECORD
                with open(record_file) as f:
                    record = f.read()
                with open(record_file, "w") as f:
                    f.write(record.replace(self.args.version_from, self.args.version_to))

                # Update build_and_package_info.py if it exists
                for root, _dirs, files in os.walk(tmp_dir):
                    if "build_and_package_info.py" in files:
                        build_info_file = os.path.join(root, "build_and_package_info.py")
                        with open(build_info_file) as f:
                            lines = f.readlines()
                        with open(build_info_file, "w") as f:
                            f.writelines(
                                f"__version__ = '{self.args.version_to}'\n" if line.startswith("__version__") else line
                                for line in lines
                            )
                        break

                # Rename dist-info directory
                new_dist_info = dist_info_file.replace(self.args.version_from, self.args.version_to)
                os.rename(
                    os.path.join(tmp_dir, dist_info_file),
                    os.path.join(tmp_dir, new_dist_info),
                )

                # Repack wheel
                subprocess.run(["wheel", "pack", tmp_dir, "-d", output_dir], check=True)

                updated_wheel_path = os.path.join(
                    output_dir,
                    os.path.basename(wheel_path.replace(self.args.version_from, self.args.version_to)),
                )

                if not os.path.exists(updated_wheel_path):
                    raise RuntimeError(f"Failed to update wheel {wheel_file}")

                logging.info(f"Version update completed for {wheel_file}, updated to {updated_wheel_path}")

    def upload_artifacts(self, distribution_dir: str) -> None:
        """Upload wheel packages using twine."""
        upload_repository = self.args.index_server_to
        username, password = self._get_credentials(upload_repository)
        is_pypi = self.credential_manager.is_pypi_index(upload_repository)

        # Get credentials from environment variables and set TWINE_USERNAME/TWINE_PASSWORD
        env = os.environ.copy()
        env["TWINE_USERNAME"] = username
        env["TWINE_PASSWORD"] = password

        cmd = [
            "twine",
            "upload",
            distribution_dir + "/*",
            "--repository",
            upload_repository,
            "--verbose",
            "--disable-progress-bar",
        ]
        if not is_pypi:
            cmd.extend(["--cert", ARTIFACTORY_CERTS_FILE, "--config-file", PYPI_RC_FILE])
        subprocess.run(cmd, check=True, env=env)


class NugetUpleveler(ArtifactUpleveler):
    """Handles NuGet package upleveling."""

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.upload_source_name = None

    @property
    def artifact_format(self) -> str:
        return "nuget"

    def _add_nuget_source(self, username: str, password: str, source_url: str, server: str, version: str) -> str:
        """Add a single NuGet source using PackageSourceCredentials environment variables."""
        if "re-artifactory-nuget-" in server:
            source_name = f"{server}-{version}"
            actual_source_url = source_url
        elif server == "testnuget":
            source_name = "testnuget.org"
            actual_source_url = "https://int.nugettest.org/api/v2/package"
        else:
            source_name = "nuget.org"
            actual_source_url = "https://api.nuget.org/v3/index.json"

        # Sanitize source name for environment variable and NuGet source name
        # (alphanumeric and underscores only)
        source_name_safe = source_name.replace("-", "_").replace(".", "_")

        # Clean up first
        subprocess.run(["nuget", "sources", "Remove", "-Name", source_name_safe], check=False)

        # Add the source using the sanitized name so it matches the environment variables
        # This allows NuGet to automatically find credentials without command-line exposure
        add_source_args = ["nuget", "sources", "Add", "-Name", source_name_safe, "-Source", actual_source_url]

        logging.info(f"Adding source {actual_source_url} with name {source_name_safe}")
        subprocess.run(add_source_args, check=True)

        return source_name_safe

    def _add_nuget_sources(self) -> list[str]:
        """Add NuGet sources and set API keys."""

        source_name_list = []
        # Add source for download
        username_from, password_from = self._get_credentials(self.args.index_server_from)
        source_name = self._add_nuget_source(
            username_from,
            password_from,
            self.url_from,
            self.args.index_server_from,
            self.args.version_from,
        )
        source_name_list.append(source_name)

        # Add source for upload
        username_to, password_to = self._get_credentials(self.args.index_server_to)
        source_name = self._add_nuget_source(
            username_to,
            password_to,
            self.url_to,
            self.args.index_server_to,
            self.args.version_to,
        )
        source_name_list.append(source_name)
        # Store the upload source name for use in upload_artifacts
        self.upload_source_name = source_name
        return source_name_list

    def _clean_up_nuget_sources(self, source_name_list: list[str]) -> None:
        """Clean up NuGet sources."""
        for source_name in source_name_list:
            logging.info(f"Removing source: {source_name}")
            subprocess.run(["nuget", "sources", "Remove", "-Name", source_name], check=True)

    def update_artifacts(self, artifact_list: list[str], input_dir: str, output_dir: str) -> None:
        """Update NuGet package versions."""
        for nuget_file in artifact_list:
            logging.info(
                f"Updating version from {self.args.version_from} to {self.args.version_to} for nuget {nuget_file}"
            )

            with tempfile.TemporaryDirectory(prefix="upload_nuget_artifact_") as tmp_dir:
                nuget_path = os.path.join(input_dir, nuget_file)

                # Create a zip file of the nuget package (NuGet packages are just zip files)
                nuget_zip_path = nuget_path.replace(".nupkg", ".zip")
                shutil.copy(nuget_path, nuget_zip_path)

                # Extract NuGet package
                with zipfile.ZipFile(nuget_zip_path, "r") as zip_ref:
                    zip_ref.extractall(tmp_dir)

                # Update .nuspec file
                nuspec_files = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if f.endswith(".nuspec")]
                if nuspec_files:
                    # Only one file in nuspec_files
                    nuspec_file = nuspec_files[0]
                    with open(nuspec_file) as f:
                        nuspec_data = f.read()
                    with open(nuspec_file, "w") as f:
                        f.write(nuspec_data.replace(self.args.version_from, self.args.version_to))

                # Update metadata file
                metadata_folder = os.path.join(tmp_dir, "package", "services", "metadata", "core-properties")
                metadata_files = [
                    os.path.join(metadata_folder, f)
                    for f in os.listdir(metadata_folder)
                    if os.path.isfile(os.path.join(metadata_folder, f))
                ]

                if metadata_files:
                    # Only one file in metadata_files
                    metadata_file = metadata_files[0]
                    with open(metadata_file) as f:
                        metadata = f.read()
                    with open(metadata_file, "w") as f:
                        f.write(metadata.replace(self.args.version_from, self.args.version_to))

                # Repack NuGet package
                updated_nuget_path = os.path.join(
                    output_dir,
                    os.path.basename(nuget_path.replace(self.args.version_from, self.args.version_to)),
                )
                updated_nuget_zip_path = updated_nuget_path.replace(".nupkg", ".zip")

                with zipfile.ZipFile(updated_nuget_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                    for root, _dirs, files in os.walk(tmp_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, tmp_dir)
                            zipf.write(file_path, arcname)

                # Create a nupkg file
                shutil.copy(updated_nuget_zip_path, updated_nuget_path)

                if not os.path.exists(updated_nuget_path):
                    raise RuntimeError(f"Failed to update nuget {nuget_file}")

                logging.info(f"Version update completed for {nuget_file}, updated to {updated_nuget_path}")

    def upload_artifacts(self, distribution_dir: str) -> None:
        """Upload NuGet packages."""
        nuget_files = [
            os.path.join(distribution_dir, f)
            for f in os.listdir(distribution_dir)
            if os.path.isfile(os.path.join(distribution_dir, f)) and f.endswith(self.artifact_suffix)
        ]

        # Use the source name that was registered in _add_nuget_sources
        source_name = self.upload_source_name
        if not source_name:
            raise RuntimeError("Upload source name not set. Ensure _add_nuget_sources was called first.")

        for nuget_file in nuget_files:
            cmd = ["nuget", "push", nuget_file, "-Source", source_name]
            subprocess.run(cmd, check=True)

    def run(self) -> None:
        """Execute NuGet upleveling with source configuration."""
        source_name_list = self._add_nuget_sources()
        super().run()
        self._clean_up_nuget_sources(source_name_list)


class ZipUpleveler(ArtifactUpleveler):
    """Handles ZIP archive upleveling."""

    @property
    def artifact_format(self) -> str:
        return "zip"

    def update_artifacts(self, artifact_list: list[str], input_dir: str, output_dir: str) -> None:
        """Update ZIP archive versions (simple copy with renamed version)."""
        for zip_file in artifact_list:
            logging.info(
                f"Updating version from {self.args.version_from} to {self.args.version_to} "
                f"for {self.artifact_format} {zip_file}"
            )

            zip_path = os.path.join(input_dir, zip_file)
            updated_zip_path = os.path.join(
                output_dir,
                os.path.basename(zip_path.replace(self.args.version_from, self.args.version_to)),
            )

            shutil.copy(zip_path, updated_zip_path)

            if not os.path.exists(updated_zip_path):
                raise RuntimeError(f"Failed to update zip {zip_file}")

            logging.info(f"Version update completed for {zip_file}, updated to {updated_zip_path}")

    def upload_artifacts(self, distribution_dir: str) -> None:
        """Upload ZIP archives using curl with netrc authentication."""
        zip_files = [
            f
            for f in os.listdir(distribution_dir)
            if os.path.isfile(os.path.join(distribution_dir, f)) and f.endswith(self.artifact_suffix)
        ]

        # Check if netrc file was provided via command line
        if self.args.netrc_file and os.path.exists(self.args.netrc_file):
            # Use the provided netrc file
            netrc_path = self.args.netrc_file
            cleanup_netrc = False
            logging.info(f"Using provided .netrc file: {netrc_path}")
        else:
            # Create temporary .netrc file with credentials
            username, password = self._get_credentials(self.args.index_server_to)
            with tempfile.NamedTemporaryFile(mode="w", delete=False, prefix="netrc_") as netrc_file:
                netrc_path = netrc_file.name
                # Extract hostname from URL
                parsed_url = urlparse(self.url_to_display)
                hostname = parsed_url.netloc

                netrc_file.write(f"machine {hostname}\n")
                netrc_file.write(f"login {username}\n")
                netrc_file.write(f"password {password}\n")

            # Set restrictive permissions on netrc file
            os.chmod(netrc_path, 0o600)
            cleanup_netrc = True
            logging.info(f"Created temporary .netrc file: {netrc_path}")

        try:
            # Use url_to_display for ZIP uploads
            for zip_file in zip_files:
                zip_path = os.path.join(distribution_dir, zip_file)
                upload_url = os.path.join(self.url_to_display, zip_file)
                cmd = [
                    "curl",
                    "-T",
                    zip_path,
                    "--cacert",
                    ARTIFACTORY_CERTS_FILE,
                    "--netrc-file",
                    netrc_path,
                    upload_url,
                ]
                subprocess.run(cmd, check=True)
        finally:
            # Clean up the temporary .netrc file only if we created it
            if cleanup_netrc and os.path.exists(netrc_path):
                os.remove(netrc_path)
                logging.info(f"Cleaned up temporary .netrc file: {netrc_path}")


class TgzUpleveler(ZipUpleveler):
    """Handles TGZ archive upleveling."""

    @property
    def artifact_format(self) -> str:
        return "tgz"


class MavenUpleveler(ArtifactUpleveler):
    """Handles Maven AAR artifact promotion and publishing.

    Dispatches on --index_server_to:
      artifactory-maven-virtual  — promote a snapshot to a release (or snapshot-to-snapshot)
                            within the internal Artifactory Maven repository.
                            Credentials: AISW_MAVEN_ARTIFACTORY_USERNAME / _PASSWORD
      maven-central       — pull an already-released artifact from index_server_from, generate
                            checksums and GPG signatures, zip into a bundle, and upload
                            to the Maven Central Portal (USER_MANAGED publishing type —
                            the final "Publish" click is done manually in the portal).

    Credentials (never in argv):
      AISW_MAVEN_ARTIFACTORY_USERNAME / _PASSWORD  — Artifactory authentication
      MAVEN_CENTRAL_BEARER_TOKEN                   — Central Portal upload token
      MAVEN_CENTRAL_GPG_PRIVATE_KEY                — ASCII-armored GPG private key
      MAVEN_CENTRAL_GPG_PASSPHRASE                 — GPG key passphrase
    """

    @property
    def artifact_format(self) -> str:
        return "maven"

    @property
    def artifact_suffix(self) -> str:
        # Maven has multiple extensions (.aar, .pom, .jar); download_artifacts is overridden.
        return ""

    # --------------------------------------------------------------------------
    # Download
    # --------------------------------------------------------------------------

    def download_artifacts(self, url: str, download_dir: str) -> list[str]:
        """Fetch Maven artifacts and save them under logical (non-timestamped) names.

        Artifactory promotion: aar + pom. maven-central: aar + pom + sources + javadoc.
        SNAPSHOT filenames in Artifactory carry a timestamp/buildNumber suffix; we
        resolve the latest by scanning the version directory, then rename on save so
        that downstream `replace(version_from, version_to)` logic works unchanged.
        """
        artifact_id = self.args.product_name
        version = self.args.version_from
        dry_run = getattr(self.args, "dry_run", False)
        to_central = self.credential_manager.is_maven_index(self.args.index_server_to)

        files: list[tuple[str, str]] = [("aar", ""), ("pom", "")]
        if to_central:
            files += [("jar", "sources"), ("jar", "javadoc")]

        remote_names = dict.fromkeys(files, "<dry-run>") if dry_run else self._resolve_remote_filenames(version, files)

        base = self._artifactory_base_url()
        downloaded: list[str] = []
        for ext, classifier in files:
            logical = f"{artifact_id}-{version}" + (f"-{classifier}" if classifier else "") + f".{ext}"
            if dry_run:
                logging.info(
                    "[dry-run] Would download %s/%s/%s/ -> %s/%s",
                    base,
                    artifact_id,
                    version,
                    download_dir,
                    logical,
                )
            else:
                file_url = f"{base}/{artifact_id}/{version}/{remote_names[(ext, classifier)]}"
                self._download_file(file_url, Path(download_dir) / logical)
            downloaded.append(logical)
        return downloaded

    def _resolve_remote_filenames(self, version: str, files: list[tuple[str, str]]) -> dict[tuple[str, str], str]:
        """Return the filename Artifactory actually serves for each (ext, classifier).

        Release versions: deterministic — `{artifact_id}-{version}[-{cls}].{ext}`.
        SNAPSHOT versions: list the version directory and pick the latest timestamped
        build matching `{artifact_id}-{base_version}-{timestamp}-{buildNumber}[-{cls}].{ext}`.
        """
        artifact_id = self.args.product_name
        if not version.endswith("-SNAPSHOT"):
            return {
                (ext, cls): f"{artifact_id}-{version}" + (f"-{cls}" if cls else "") + f".{ext}" for ext, cls in files
            }

        base_version = version.removesuffix("-SNAPSHOT")
        dir_url = f"{self._artifactory_base_url()}/{artifact_id}/{version}/"
        username, password = self._get_credentials(self.args.index_server_from)
        resp = requests.get(
            dir_url,
            auth=HTTPBasicAuth(username, password),
            verify=ARTIFACTORY_CERTS_FILE,
            timeout=120,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to list {dir_url}: HTTP {resp.status_code}")

        # Extract filenames from Artifactory's HTML directory listing via href values.
        candidates = [
            m.group(1) for m in (re.search(r'href=["\']([^"\']+)["\']', line) for line in resp.text.splitlines()) if m
        ]

        result: dict[tuple[str, str], str] = {}
        for ext, classifier in files:
            cls_part = f"-{re.escape(classifier)}" if classifier else ""
            pattern = re.compile(
                rf"^{re.escape(artifact_id)}-{re.escape(base_version)}"
                rf"-(\d{{8}}\.\d{{6}})-(\d+){cls_part}\.{re.escape(ext)}$"
            )
            matches: list[tuple[str, int, str]] = []
            for name in candidates:
                m = pattern.match(name)
                if m:
                    matches.append((m.group(1), int(m.group(2)), name))
            if not matches:
                raise RuntimeError(
                    f"No SNAPSHOT file under {dir_url} matches classifier={classifier!r} extension={ext!r}"
                )
            matches.sort(key=lambda t: (t[0], t[1]))
            result[(ext, classifier)] = matches[-1][2]
        return result

    # --------------------------------------------------------------------------
    # Update (artifactory promotion only; maven-central enforces v_from == v_to)
    # --------------------------------------------------------------------------

    def update_artifacts(self, artifact_list: list[str], input_dir: str, output_dir: str) -> None:
        """Rename to version_to and rewrite the POM's <version>."""
        artifact_id = self.args.product_name
        v_from, v_to = self.args.version_from, self.args.version_to
        dry_run = getattr(self.args, "dry_run", False)

        for filename in artifact_list:
            new_name = filename.replace(f"{artifact_id}-{v_from}", f"{artifact_id}-{v_to}")
            src = Path(input_dir) / filename
            dst = Path(output_dir) / new_name
            if dry_run:
                logging.info("[dry-run] Would produce %s from %s", dst, src)
                continue
            shutil.copy2(src, dst)
            if dst.suffix == ".pom":
                maven_publish_utils.rewrite_pom_version(dst, v_to)

    # --------------------------------------------------------------------------
    # Upload (dispatches on destination)
    # --------------------------------------------------------------------------

    def upload_artifacts(self, distribution_dir: str) -> None:
        group_id = "com.qualcomm.qti"
        artifact_id = self.args.product_name
        if self.credential_manager.is_maven_index(self.args.index_server_to):
            self._upload_maven_central(distribution_dir, group_id, artifact_id)
        else:
            self._upload_artifactory(distribution_dir, group_id, artifact_id)

    # --------------------------------------------------------------------------
    # Private helpers
    # --------------------------------------------------------------------------

    def _artifactory_base_url(self) -> str:
        config = ConfigParser()
        config.read(INI_FILE)
        return config.get(self.args.index_server_from, "repository").rstrip("/")

    def _artifactory_upload_base_url(self) -> str:
        config = ConfigParser()
        config.read(INI_FILE)
        return config.get(self.args.index_server_to, "repository").rstrip("/")

    def _download_file(self, url: str, dest: Path) -> None:
        username, password = self._get_credentials(self.args.index_server_from)
        resp = requests.get(
            url,
            auth=HTTPBasicAuth(username, password),
            verify=ARTIFACTORY_CERTS_FILE,
            timeout=120,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to download {url}: HTTP {resp.status_code}")
        dest.write_bytes(resp.content)
        logging.info("Downloaded: %s -> %s", url, dest)

    def _upload_artifactory(self, distribution_dir: str, group_id: str, artifact_id: str) -> None:
        """Deploy the prepared aar + pom (plus dummy sources/javadoc jars) via mvn deploy-file."""
        version_to = self.args.version_to
        dry_run = getattr(self.args, "dry_run", False)

        username, password = self._get_credentials(self.args.index_server_to)
        group_path = group_id.replace(".", "/")
        upload_base = self._artifactory_upload_base_url()
        if not upload_base.endswith(f"/{group_path}"):
            raise RuntimeError(
                f"Upload base URL {upload_base!r} does not end with /{group_path}; "
                f"check config.ini [artifactory-maven-virtual]"
            )
        repo_url = upload_base[: -len(f"/{group_path}")]
        repo_id = "snapshots" if version_to.endswith("-SNAPSHOT") else "releases"

        logging.info(
            "Deploying %s:%s:%s -> %s (%s repo) in Artifactory",
            group_id,
            artifact_id,
            version_to,
            repo_url,
            repo_id,
        )

        dist = Path(distribution_dir)
        aar = dist / f"{artifact_id}-{version_to}.aar"
        pom = dist / f"{artifact_id}-{version_to}.pom"

        jar_dir = dist / "jars"
        jar_dir.mkdir(exist_ok=True)
        sources_jar, javadoc_jar = maven_publish_utils.generate_dummy_jars(
            jar_dir,
            group_id,
            artifact_id,
            version_to,
            dry_run=dry_run,
        )

        with (
            maven_publish_utils.render_settings_xml(username, password, repo_url) as settings_xml,
        ):
            ssl_opts, truststore = maven_publish_utils.qualcomm_ssl_opts()
            try:
                maven_publish_utils.mvn_deploy_file(
                    aar=aar,
                    pom=pom,
                    sources_jar=sources_jar,
                    javadoc_jar=javadoc_jar,
                    group_id=group_id,
                    artifact_id=artifact_id,
                    version=version_to,
                    repository_id=repo_id,
                    repository_url=repo_url,
                    settings_xml=settings_xml,
                    dry_run=dry_run,
                    maven_local_repo=jar_dir,
                    ssl_opts=ssl_opts,
                )
            finally:
                if truststore:
                    with contextlib.suppress(FileNotFoundError):
                        truststore.unlink()

    def _upload_maven_central(self, distribution_dir: str, group_id: str, artifact_id: str) -> None:
        """Bundle, checksum, GPG-sign, zip, and upload to the Maven Central Portal."""
        if self.credential_manager.is_maven_index(self.args.index_server_from):
            raise ValueError(
                "--index_server_from cannot be 'maven-central' when publishing to maven-central; "
                "provide the Artifactory server to download the release artifacts from."
            )
        if self.args.version_from != self.args.version_to:
            raise ValueError(
                f"maven-central publishing requires version_from == version_to; "
                f"got {self.args.version_from!r} vs {self.args.version_to!r}"
            )
        version = self.args.version_from
        dry_run = getattr(self.args, "dry_run", False)

        bearer_token = os.environ.get("MAVEN_CENTRAL_BEARER_TOKEN", "")
        gpg_private_key = os.environ.get("MAVEN_CENTRAL_GPG_PRIVATE_KEY", "")
        gpg_passphrase = os.environ.get("MAVEN_CENTRAL_GPG_PASSPHRASE", "")

        if not dry_run:
            if not bearer_token:
                raise RuntimeError("MAVEN_CENTRAL_BEARER_TOKEN is not set")
            if not gpg_private_key:
                raise RuntimeError("MAVEN_CENTRAL_GPG_PRIVATE_KEY is not set")
            if not gpg_passphrase:
                raise RuntimeError("MAVEN_CENTRAL_GPG_PASSPHRASE is not set")

        dist = Path(distribution_dir)
        aar = dist / f"{artifact_id}-{version}.aar"
        pom = dist / f"{artifact_id}-{version}.pom"
        sources_jar = dist / f"{artifact_id}-{version}-sources.jar"
        javadoc_jar = dist / f"{artifact_id}-{version}-javadoc.jar"

        if dry_run:
            logging.info(
                "[dry-run] Would arrange com/qualcomm/qti/%s/%s/ bundle, "
                "run `mvn -f checksumpom.xml package` to generate .md5/.sha1, "
                "GPG-sign each .aar/.pom/.jar -> .asc, "
                "zip into <artifact_id>-<version>.zip, "
                "and upload to https://central.sonatype.com/api/v1/publisher/upload"
                "?publishingType=USER_MANAGED",
                artifact_id,
                version,
            )
            return

        # 1. Arrange into the Maven Central directory convention:
        #    com/qualcomm/qti/{artifact_id}/{version}/
        bundle_root = dist / "bundle"
        files_dir = bundle_root / "com" / "qualcomm" / "qti" / artifact_id / version
        files_dir.mkdir(parents=True)

        for src in [aar, pom, sources_jar, javadoc_jar]:
            shutil.copy2(src, files_dir / src.name)

        # 2. Generate checksums (.md5, .sha1) via checksumpom.xml
        # checksumpom.xml reads from a `files` sub-directory
        checksum_files_dir = dist / "checksum" / "files"
        checksum_files_dir.mkdir(parents=True)
        for src in [aar, pom, sources_jar, javadoc_jar]:
            shutil.copy2(src, checksum_files_dir / src.name)

        maven_publish_utils.generate_checksums(
            checksum_files_dir,
            group_id,
            artifact_id,
            version,
            dry_run=dry_run,
        )
        # Copy checksums into the bundle directory
        for checksum_file in checksum_files_dir.glob("*.md5"):
            shutil.copy2(checksum_file, files_dir / checksum_file.name)
        for checksum_file in checksum_files_dir.glob("*.sha1"):
            shutil.copy2(checksum_file, files_dir / checksum_file.name)

        # 3. GPG-sign every .aar / .pom / .jar in the bundle directory
        for artifact_file in sorted(files_dir.iterdir()):
            if artifact_file.suffix in (".aar", ".pom", ".jar"):
                maven_publish_utils.sign_file_gpg(
                    artifact_file,
                    gpg_private_key,
                    gpg_passphrase,
                    dry_run=dry_run,
                )

        # 4. Zip the bundle
        bundle_zip = dist / f"{artifact_id}-{version}.zip"
        shutil.make_archive(
            str(bundle_zip.with_suffix("")),
            "zip",
            root_dir=str(bundle_root),
            base_dir="com",
        )

        # 5. Upload to Maven Central Portal
        deployment_id = maven_publish_utils.upload_to_maven_central(
            bundle_zip,
            bearer_token,
            dry_run=dry_run,
        )
        logging.info("Bundle staged at Maven Central. Deployment ID: %s", deployment_id)
        logging.info(
            "Final step: visit https://central.sonatype.com/publishing/deployments "
            "and click 'Publish' to promote to Maven Central."
        )


class UplevelingFactory:
    """Factory class to create appropriate upleveler instances."""

    _uplevelers: ClassVar[dict[str, ArtifactUpleveler]] = {
        "wheel": WheelUpleveler,
        "nuget": NugetUpleveler,
        "zip": ZipUpleveler,
        "tgz": TgzUpleveler,
        "maven": MavenUpleveler,
    }

    @classmethod
    def create_upleveler(cls, args: argparse.Namespace) -> ArtifactUpleveler:
        """Create and return the appropriate upleveler instance."""
        upleveler_class = cls._uplevelers.get(args.artifact_format)
        if not upleveler_class:
            raise ValueError(
                f"Invalid artifact format: {args.artifact_format}. Must be one of {list(cls._uplevelers.keys())}"
            )
        return upleveler_class(args)


def parse_arguments() -> argparse.Namespace:
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(
        description="Upload artifacts with version upleveling support.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--product_name",
        type=str,
        required=True,
        help="Product name. Default is onnxruntime-qnn",
    )
    parser.add_argument(
        "--artifact_format",
        type=str,
        required=True,
        choices=["wheel", "nuget", "zip", "tgz", "maven"],
        help="The format of artifact. Choose one of [wheel, nuget, zip, tgz, maven].",
    )
    parser.add_argument(
        "--version_from",
        type=str,
        required=True,
        help="Source version of artifact."
        "For python wheel, the format should be <version>.<version>.<version><.><suffix>."
        "And the suffix should be <letters><numbers>."
        "For nuget package, the format should be <version>.<version>.<version>-<suffix>."
        "Please note that underscore (_) is not allowed in the version format.",
    )
    parser.add_argument(
        "--version_to",
        type=str,
        default="",
        help="Target version of artifact (defaults to version_from if not specified)."
        "For python wheel, the format should be <version>.<version>.<version><.><suffix>."
        "And the suffix should be <letters><numbers>."
        "For nuget package, the format should be <version>.<version>.<version>-<suffix>."
        "Please note that underscore (_) is not allowed in the version format.",
    )
    parser.add_argument(
        "--index_server_from",
        type=str,
        required=True,
        help="Source server. Choose one of ["
        "pypi, testpypi, nuget, testnuget, "
        "test-users, test-project, project, public]",
    )
    parser.add_argument(
        "--index_server_to",
        type=str,
        required=True,
        help="Target server.Choose one of ["
        "pypi, testpypi, nuget, testnuget, "
        "test-users, test-project, project, public]",
    )
    parser.add_argument(
        "--netrc_file",
        type=str,
        default="",
        help="Path to .netrc file for curl authentication (optional, only used for zip and tgz uploads).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing (Maven format only).",
    )

    args = parser.parse_args()

    # Set version_to to version_from if not specified
    if not args.version_to:
        args.version_to = args.version_from

    # Transform index server names with appropriate prefixes.
    # Maven uses literal server names (e.g. artifactory-maven-virtual, maven-central);
    # skip prefix transformation for the maven format.
    valid_artifactory_suffixes = {"test-users", "test-project", "project", "public"}

    if args.artifact_format in ARTIFACTORY_PREFIXES:
        prefix = ARTIFACTORY_PREFIXES[args.artifact_format]

        if args.index_server_from in valid_artifactory_suffixes:
            args.index_server_from = f"{prefix}-{args.index_server_from}"

        if args.index_server_to in valid_artifactory_suffixes:
            args.index_server_to = f"{prefix}-{args.index_server_to}"

    return args


def main():
    """Main entry point for the upleveling script."""
    args = parse_arguments()

    # Create appropriate upleveler and run
    upleveler = UplevelingFactory.create_upleveler(args)
    upleveler.run()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()
