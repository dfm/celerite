set(CMAKE_CXX_STANDARD 11)
set( CMAKE_CXX_EXTENSIONS OFF)

macro (CELERITE_EXEC NAME)
    add_executable(${NAME} ${NAME}.cc)
    target_link_libraries(${NAME} ${CELERITE_LIBRARIES})
endmacro (CELERITE_EXEC)

celerite_exec(benchmark)
celerite_exec(carma_comp)

macro (CELERITE_TEST NAME)
    celerite_exec(${NAME})
    add_test(${NAME} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${NAME})
endmacro (CELERITE_TEST)

celerite_test(test_solvers)
celerite_test(test_poly)
