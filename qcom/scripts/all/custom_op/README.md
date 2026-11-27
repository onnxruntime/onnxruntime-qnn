# HTP Custom Op tests

Run `sh custom_op/run_custom_op_tests.sh` from an unpacked Linux x86_64 or
Android aarch64 test archive. No Python, LLVM, Hexagon SDK, or op-package
generator is needed on the test host. The archive contains the matching ORT/QNN
libraries and provider test executable alongside this directory.

The script tests JIT inference and domain registration, loads the packaged
Linux-generated EPContext in a separate process without
`ORT_QNN_CUSTOM_OP_DOMAINS`, and generates and reloads a fresh native context.
Any failed or skipped test fails the script. XML reports are written beside
the provider test executable.

`contexts/73` targets `htp_arch=73`, `soc_model=60`, `vtcm_mb=8`.
`contexts/81` targets `htp_arch=81`, `soc_model=87`, `vtcm_mb=8`.
Each contains the source QDQ model, an EPContext model, its external binary,
and target metadata. Linux archives test all targets on the HTP simulator.
Android archives also include v75 (`soc_model=57`) and v79 (`soc_model=69`),
with `vtcm_mb=8`, for the device CI matrix. The script queries the device
architecture and selects its matching context and DSP library. JIT and generation use both host and DSP package halves;
pure EPContext consumption registers only the DSP half.

Contexts are produced with the archive build's native Linux provider tests and
QNN SDK. Packages and context binaries must stay together across SDK upgrades.
Windows ARM64 AOT consumption and skel signing remain follow-up work.
