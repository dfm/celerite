macro (CELERITE_EXEC NAME)
    add_executable(${NAME} ${NAME}.cc)
endmacro (CELERITE_EXEC)

celerite_exec(benchmark)
celerite_exec(carma_comp)

macro (CELERITE_TEST NAME)
    celerite_exec(${NAME})
    add_test(${NAME} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${NAME})
endmacro (CELERITE_TEST)

celerite_test(test_solvers)
# celerite_test(test_gp)
# celerite_test(test_gp_grad)
# celerite_test(test_grad)
celerite_test(test_poly)
