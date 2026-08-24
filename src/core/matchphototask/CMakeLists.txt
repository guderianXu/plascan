cmake_minimum_required(VERSION 3.16)
project(plascan_matchphototask LANGUAGES CXX)

add_library(matchphototask STATIC
    algorithm/MatchPhotosAlgorithmPlan.cpp
    algorithm/MatchPhotosAlgorithmPlan.h
    algorithm/MatchPhotosAlgorithmSelector.cpp
    algorithm/MatchPhotosAlgorithmSelector.h
    pair_selection/PairSelectionPolicy.cpp
    pair_selection/PairSelectionPolicy.h
    pair_selection/PairSelector.cpp
    pair_selection/PairSelector.h
    pair_selection/SparseSceneOverlapAnalyzer.cpp
    pair_selection/SparseSceneOverlapAnalyzer.h
    pair_selection/PairTypes.cpp
    pair_selection/PairTypes.h
    runtime/MatchPhotosRuntime.cpp
    runtime/MatchPhotosRuntime.h
    runtime/MatchPhotosFeatureCache.cpp
    runtime/MatchPhotosFeatureCache.h
    runtime/MatchPhotosMaskSupport.cpp
    runtime/MatchPhotosMaskSupport.h
    runtime/MatchPhotosParallelism.cpp
    runtime/MatchPhotosParallelism.h
    stages/FeaturePreparationQueue.cpp
    stages/FeaturePreparationQueue.h
    stages/FeatureStage.cpp
    stages/FeatureStage.h
    stages/GeometryVerifyStage.cpp
    stages/GeometryVerifyStage.h
    stages/GuidedMatchStage.cpp
    stages/GuidedMatchStage.h
    stages/GuidedMatchPolicy.cpp
    stages/GuidedMatchPolicy.h
    stages/MatchingStage.cpp
    stages/MatchingStage.h
    stages/ReferencePoseEpipolarGeometry.cpp
    stages/ReferencePoseEpipolarGeometry.h
    stages/TrackBuildStage.cpp
    stages/TrackBuildStage.h
    tie_points/TiePointTrackManager.cpp
    tie_points/TiePointTrackManager.h
    task/MatchPhotosContext.h
    task/MatchPhotosOptions.h
    task/MatchPhotosResult.h
    task/MatchPhotosTask.cpp
    task/MatchPhotosTask.h
)

target_include_directories(matchphototask PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/algorithm
    ${CMAKE_CURRENT_SOURCE_DIR}/pair_selection
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime
    ${CMAKE_CURRENT_SOURCE_DIR}/stages
    ${CMAKE_CURRENT_SOURCE_DIR}/tie_points
    ${CMAKE_CURRENT_SOURCE_DIR}/task
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../overlap
    ${CMAKE_CURRENT_SOURCE_DIR}/../image_matching
    ${CMAKE_CURRENT_SOURCE_DIR}/../sfm
)

target_link_libraries(matchphototask PUBLIC
    Qt6::Core
    overlap
    image_matching
    plascan_common_io
    plascan_common_model
    plascan_common_project
    sfm
)

target_link_libraries(matchphototask PRIVATE
    plascan_common_concurrency
)

target_compile_definitions(matchphototask PRIVATE PLASCAN_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_compile_features(matchphototask PUBLIC cxx_std_20)
