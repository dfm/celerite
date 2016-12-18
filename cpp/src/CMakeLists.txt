macro (GENRP_EXEC NAME)
    add_executable(${NAME} ${NAME}.cc)
endmacro (GENRP_EXEC)

genrp_exec(benchmark)
genrp_exec(carma_comp)

macro (GENRP_TEST NAME)
    genrp_exec(${NAME})
    add_test(${NAME} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${NAME})
endmacro (GENRP_TEST)

genrp_test(test_solvers)
genrp_test(test_gp)
genrp_test(test_gp_grad)
genrp_test(test_grad)
genrp_test(test_poly)
