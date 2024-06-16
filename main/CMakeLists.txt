get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRArithDialect
        MLIROptLib
        )

add_llvm_executable(main 
        main.cpp 
        Dialect.cpp
        Combine.cpp

        DEPENDS
        CIMOpsIncGen)
# add_dependencies(main )

llvm_update_compile_flags(main)
target_link_libraries(main PRIVATE ${LIBS})

mlir_check_all_link_libraries(main)
