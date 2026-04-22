VERS_1.0 {
  # Export all symbols so that unit tests can call EP-internal functions directly.
  # This file is only used for coverage builds (ENABLE_COVERAGE=1).
  # Do NOT use in production builds.
  global:
    *;
};
