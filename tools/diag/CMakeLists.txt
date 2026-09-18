cmake_minimum_required(VERSION 3.16)

project(Diag LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(
  DEAL_II_DIR
  "/opt/dealii-9.7.1-codim/lib/cmake/deal.II"
  CACHE PATH
  "Path containing deal.IIConfig.cmake"
)

find_package(deal.II 9.7.1 REQUIRED HINTS "${DEAL_II_DIR}" "$ENV{DEAL_II_DIR}")
deal_ii_initialize_cached_variables()

add_executable(diag diag.cpp)
deal_ii_setup_target(diag)
