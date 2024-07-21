get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRArithDialect
        MLIROptLib
        MLIRFuncAllExtensions
        )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frtti")
add_llvm_executable(main 
        main.cpp 
        Dialect.cpp
        CIMISADialect.cpp
        Combine.cpp
        Parser.cpp
        pass/ShapeInferencePass.cpp
        pass/MemoryAddressAllocationPass.cpp
        pass/DecomposeAffineOpPass.cpp
        pass/FoldMemRefAliasPass.cpp
        pass/CIMLoweringPass.cpp
        pass/CIMBranchConvertPass.cpp
        pass/CodeGenPass.cpp

        DEPENDS
        CIMOpsIncGen
        CIMShapeInferenceInterfaceIncGen
        CIMISAOpsIncGen)
# add_dependencies(main )

set(LLVM_ENABLE_EH ON)
message(STATUS "The value of LLVM_COMPILE_FLAGS 1 is: ${LLVM_COMPILE_FLAGS}")
llvm_update_compile_flags(main)
message(STATUS "The value of LLVM_COMPILE_FLAGS 2 is: ${LLVM_COMPILE_FLAGS}")
target_link_libraries(main PRIVATE ${LIBS})
message(STATUS "The value of LLVM_COMPILE_FLAGS 3 is: ${LLVM_COMPILE_FLAGS}")
mlir_check_all_link_libraries(main)
message(STATUS "The value of LLVM_COMPILE_FLAGS 4 is: ${LLVM_COMPILE_FLAGS}")
