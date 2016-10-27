macro (GENRP_TEST NAME)
    add_executable(${NAME} ${NAME}.cc)
    target_link_libraries(${NAME} ${GENRP_LIBRARIES})
    add_test(${NAME} ${NAME}
             COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${NAME})
endmacro (GENRP_TEST)

genrp_test(test_solvers)
