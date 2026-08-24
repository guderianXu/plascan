include_guard(GLOBAL)

function(plascan_verify_source_checkout dependency_name source_dir expected_commit marker)
  if(NOT EXISTS "${source_dir}/${marker}")
    message(FATAL_ERROR
      "${dependency_name} source is incomplete: ${source_dir}/${marker}. "
      "Initialize the corresponding source submodule listed in README.md")
  endif()

  find_package(Git QUIET)
  if(NOT Git_FOUND)
    message(FATAL_ERROR
      "Git is required to verify the pinned ${dependency_name} source checkout")
  endif()
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
    WORKING_DIRECTORY "${source_dir}"
    RESULT_VARIABLE _plascan_git_result
    OUTPUT_VARIABLE _plascan_actual_commit
    ERROR_VARIABLE _plascan_git_error
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _plascan_git_result EQUAL 0)
    message(FATAL_ERROR
      "Cannot verify ${dependency_name} at ${source_dir}: ${_plascan_git_error}")
  endif()
  if(NOT _plascan_actual_commit STREQUAL expected_commit)
    message(FATAL_ERROR
      "${dependency_name} is not at the pinned commit. Expected ${expected_commit}, "
      "found ${_plascan_actual_commit} in ${source_dir}")
  endif()
endfunction()
