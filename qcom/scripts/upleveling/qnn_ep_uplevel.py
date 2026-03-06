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
Supports Python wheels, NuGet packages, and ZIP archives.
"""

import argparse
import logging
import os
import shutil
import ssl
import subprocess
import tempfile
import zipfile
from abc import ABC, abstractmethod
from configparser import ConfigParser
from typing import ClassVar
from urllib.parse import urlparse

import requests
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
}

ARTIFACT_SUFFIXES = {"wheel": ".whl", "nuget": ".nupkg", "zip": ".zip"}


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

        # Extract artifact file links
        artifact_list = [line.split('"')[1] for line in response.text.splitlines() if self.artifact_suffix in line]

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
            logging.info(f"Updating version from {self.args.version_from} to {self.args.version_to} for zip {zip_file}")

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


class UplevelingFactory:
    """Factory class to create appropriate upleveler instances."""

    _uplevelers: ClassVar[dict[str, ArtifactUpleveler]] = {
        "wheel": WheelUpleveler,
        "nuget": NugetUpleveler,
        "zip": ZipUpleveler,
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
        choices=["wheel", "nuget", "zip"],
        help="The format of artifact. Choose one of [wheel, nuget, zip].",
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
        help="Path to .netrc file for curl authentication (optional, only used for zip uploads).",
    )

    args = parser.parse_args()

    # Set version_to to version_from if not specified
    if not args.version_to:
        args.version_to = args.version_from

    # Transform index server names with appropriate prefixes
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
