add_library(overlap STATIC
    GroundBackProjector.cpp
    OverlapAnalyzer.cpp
    VocabularyOverlapRetriever.cpp
    VocabularyOverlapRetriever.h
)

target_include_directories(overlap PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../camera
    ${OpenCV_INCLUDE_DIRS}
)

target_include_directories(overlap PRIVATE
    ${CMAKE_SOURCE_DIR}/src/common
)

target_link_libraries(overlap PUBLIC
    camera
    ${OpenCV_LIBS}
    $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)
