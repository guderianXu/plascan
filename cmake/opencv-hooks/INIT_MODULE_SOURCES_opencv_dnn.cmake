# OpenCV 5.0.0 detects cl.exe as an ASM compiler on Windows, but its vendored
# x86-64 MLAS kernels use GNU .S syntax. CMake consequently creates object
# rules that do not produce linkable files. Remove that optional object target
# from DNN and let OpenCV use its built-in SGEMM implementation on MSVC.
if(MSVC AND TARGET opencv_dnn_mlas)
  list(FILTER OPENCV_MODULE_opencv_dnn_SOURCES EXCLUDE REGEX "opencv_dnn_mlas")
  remove_definitions(-DHAVE_MLAS=1)
  set_property(TARGET opencv_dnn_mlas PROPERTY EXCLUDE_FROM_ALL TRUE)
  set_property(TARGET opencv_dnn_mlas PROPERTY MSVC_RUNTIME_LIBRARY "")
  target_compile_options(opencv_dnn_mlas PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:/MD$<$<CONFIG:Debug>:d>>")
  set(HAVE_MLAS 0)
  set(OPENCV_DNN_MLAS_ENABLED 0 CACHE INTERNAL "" FORCE)
  set(OPENCV_DNN_MLAS_SKIP_REASON
    "GNU-style MLAS assembly is not supported by MSVC"
    CACHE INTERNAL "" FORCE)
endif()
