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
        passes/ShapeInferencePass.cpp
        passes/MemoryAddressAllocationPass.cpp
        passes/DecomposeAffineOpPass.cpp
        passes/FoldMemRefAliasPass.cpp
        passes/CIMLoweringPass.cpp
        passes/CIMBranchConvertPass.cpp
        passes/ExtractAddressComputationPass.cpp
        passes/RR2RIPass.cpp
        passes/ConstantExpandPass.cpp
        passes/LoopUnrollPass.cpp
        passes/CastEliminationPass.cpp
        passes/CommonSubexpressionExposePass.cpp
        codegen/CodeGenPass.cpp
        codegen/InstructionWriter.cpp
        codegen/CIMFlowInstructionWriter.cpp

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
target_link_libraries(main PRIVATE glog)
