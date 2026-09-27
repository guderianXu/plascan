# Compatibility target for PlaScan modules that consume PlaCamera components.
add_library(camera INTERFACE)
target_link_libraries(camera
    INTERFACE placamera::placamera
              placamera::state
              placamera::gdal
)
