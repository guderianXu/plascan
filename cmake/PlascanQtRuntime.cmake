# Qt 的网络 TLS 后端以运行时插件形式加载，仅链接 Qt6::Network 并不会自动把插件复制到
# 可执行文件目录。该函数统一处理开发构建中的插件部署，安装阶段再由现有打包脚本把
# `<runtime>/tls` 复制到安装包的 `plugins/tls`。
function(plascan_deploy_qt_tls_runtime target_name)
  if(NOT WIN32)
    return()
  endif()

  if(NOT TARGET "${target_name}")
    message(FATAL_ERROR
      "plascan_deploy_qt_tls_runtime: target does not exist: ${target_name}")
  endif()

  set(_plascan_tls_backend_targets)
  foreach(_plascan_tls_target IN ITEMS
      Qt6::QSchannelBackendPlugin
      Qt6::QOpenSSLBackendPlugin)
    if(TARGET "${_plascan_tls_target}")
      list(APPEND _plascan_tls_backend_targets "${_plascan_tls_target}")
    endif()
  endforeach()

  if(NOT _plascan_tls_backend_targets)
    message(FATAL_ERROR
      "PlaScan requires a functional Qt TLS backend for HTTPS model downloads. "
      "Rebuild Qt Network with Schannel or OpenSSL support.")
  endif()

  # qcertonlybackend 只提供证书解析能力，不是独立的 TLS 传输后端；存在时仍一起部署，
  # 使 Qt 的证书处理能力与 vcpkg 提供的标准运行时保持一致。
  set(_plascan_tls_plugin_targets ${_plascan_tls_backend_targets})
  if(TARGET Qt6::QCertOnlyBackendPlugin)
    list(APPEND _plascan_tls_plugin_targets Qt6::QCertOnlyBackendPlugin)
  endif()

  add_custom_command(TARGET "${target_name}" POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E make_directory
      "$<TARGET_FILE_DIR:${target_name}>/tls"
    COMMENT "Deploying Qt TLS runtime plugins for ${target_name}")

  foreach(_plascan_tls_target IN LISTS _plascan_tls_plugin_targets)
    add_custom_command(TARGET "${target_name}" POST_BUILD
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "$<TARGET_FILE:${_plascan_tls_target}>"
        "$<TARGET_FILE_DIR:${target_name}>/tls/"
      VERBATIM)
  endforeach()
endfunction()
