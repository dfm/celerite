macro (GENRP_EXEC NAME)
    add_executable(${NAME} ${NAME}.cc)
endmacro (GENRP_EXEC)

genrp_exec(benchmark)

macro (GENRP_TEST NAME)
    genrp_exec(${NAME})
    add_test(${NAME} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${NAME})
endmacro (GENRP_TEST)

genrp_test(test_solvers)
genrp_test(test_gp)
