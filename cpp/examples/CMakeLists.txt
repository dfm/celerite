macro (GENRP_TEST NAME)
    add_executable(demo demo.cpp)
    add_test(NAME ${NAME}
             COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${NAME})
endmacro (GENRP_TEST)

genrp_test(demo)
