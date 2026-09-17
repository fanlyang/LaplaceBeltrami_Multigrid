# Build and run the solver without installing deal.II locally.
#
#   docker build -t laplace-beltrami .
#   docker run --rm laplace-beltrami                 # degree 3, 5 cycles
#   docker run --rm laplace-beltrami 4 4             # degree 4, 4 cycles
#
# The base image is a deal.II 9.7.1 build that additionally instantiates the
# multigrid classes for dim != spacedim. A stock dealii/dealii image links
# against no such symbols, which is why this surface (codimension one)
# problem needs the codim build.

FROM fanyoung/dealii-codim:9.7.1

WORKDIR /home/dealii/solver

COPY CMakeLists.txt ./
COPY src ./src

RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build -j"$(nproc)"

# Default arguments: polynomial degree, number of refinement cycles.
ENTRYPOINT ["./build/solver"]
CMD ["3", "5"]
