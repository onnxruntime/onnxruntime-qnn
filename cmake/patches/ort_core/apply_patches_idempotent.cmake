# apply_patches_idempotent.cmake
#
# Applies each patch in PATCH_FILES (a cmake semicolon-separated list) to WORKING_DIR using
# patch -N (--forward). execute_process does not propagate non-zero exit codes, so an
# already-applied patch (exit 1) is silently skipped — making the step safe to run on every build
# regardless of whether the patches were already applied by FetchContent's PATCH_COMMAND.
#
# Variables (passed via -D):
#   PATCH_EXECUTABLE  — path to the patch tool
#   PATCH_FILES       — cmake list of absolute patch file paths (sorted)
#   WORKING_DIR       — directory to apply patches in (ort_core_SOURCE_DIR)

foreach(_patch IN LISTS PATCH_FILES)
  get_filename_component(_patch_name "${_patch}" NAME)
  message(STATUS "Applying patch (idempotent): ${_patch_name}")
  execute_process(
    COMMAND ${PATCH_EXECUTABLE} --binary --ignore-whitespace -p1 -N -i "${_patch}"
    WORKING_DIRECTORY "${WORKING_DIR}"
  )
endforeach()
