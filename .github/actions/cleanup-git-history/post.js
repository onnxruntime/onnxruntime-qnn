// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

const fs = require("fs");
const path = require("path");

const workspace = process.env.STATE_workspace;
const workspaceRoot = process.env.STATE_workspace_root || process.env.RUNNER_WORKSPACE;

// Refuse a malformed environment rather than risking deletion outside the
// runner's work directory. actions/checkout uses this default workspace path.
if (!workspace || !workspaceRoot) {
  console.warn("Git cleanup skipped: GITHUB_WORKSPACE or RUNNER_WORKSPACE is unavailable.");
  process.exit(0);
}

const resolvedWorkspace = path.resolve(workspace);
const resolvedRoot = path.resolve(workspaceRoot);
if (resolvedWorkspace === resolvedRoot || !resolvedWorkspace.startsWith(`${resolvedRoot}${path.sep}`)) {
  console.error(`Git cleanup refused unsafe workspace path: ${resolvedWorkspace}`);
  process.exit(1);
}

const gitDir = path.join(resolvedWorkspace, ".git");
try {
  // Deleting .git, instead of running git gc, is deliberate: a full-history
  // checkout keeps all commits reachable, so gc cannot recover that space.
  fs.rmSync(gitDir, { recursive: true, force: true, maxRetries: 3, retryDelay: 250 });
  console.info(`Removed Git database: ${gitDir}`);
} catch (error) {
  console.error(`Failed to remove Git database ${gitDir}: ${error.message}`);
  process.exitCode = 1;
}
