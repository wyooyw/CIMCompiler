#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>

#include "cim/Parser.h"
#include "cim/Dialect.h"



static const boost::property_tree::ptree&  get_item(const boost::property_tree::ptree& ast, int index){
    auto it = ast.begin();
    std::advance(it, index);
    return it->second;
}

std::string MLIRGenImpl::safe_get_str(const boost::property_tree::ptree& ast, const std::string& key){
    if (ast.count(key)){
        return ast.get<std::string>(key);
    }else{
        // tell user
        mlir::emitError(mlir::UnknownLoc::get(builder.getContext()),
                "[safe_get_str] Key error: " + key);
        std::exit(1);
        return "";
    }
}

const boost::property_tree::ptree& MLIRGenImpl::safe_get_child(const boost::property_tree::ptree& ast, const std::string& key){
    if (ast.count(key)){
        return ast.get_child(key);
    }else{
        // tell user
        mlir::emitError(mlir::UnknownLoc::get(builder.getContext()),
                "[safe_get_child] Key error: " + key);
        std::exit(1);
        return ast;
    }
}

    MLIRGenImpl::MLIRGenImpl(mlir::MLIRContext &context) : builder(&context), loc(builder.getUnknownLoc()) {

        GLOBAL_MEMORY = builder.getStringAttr("global");
        LOCAL_MEMORY = builder.getStringAttr("local");
        MACRO_MEMORY = builder.getStringAttr("macro");
        // loc = builder.getUnknownLoc();

    }

    mlir::ModuleOp MLIRGenImpl::parseJson(std::string json_path) { 
        // Parse the module
        boost::property_tree::ptree ast;
        boost::property_tree::read_json(json_path, ast);
        return parseModule(ast);
    }

    mlir::ModuleOp MLIRGenImpl::parseModule(const boost::property_tree::ptree& ast) { 
        // Parse the module
        mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());
        
        
        auto program = safe_get_child(ast, "program");
        auto define_func_list = get_item(program, 0);

        mlir::Block *module_body = module.getBody();
        block_stack.push(module_body);
        builder.setInsertionPointToEnd(module_body);

        for (const auto& pair : program) {
            auto ast_define_func = safe_get_child(pair.second, "define_function");
            parse_func(ast_define_func);
        }

        block_stack.pop();
        return module;
    }

    void MLIRGenImpl::init_func_in_sign_table(const std::string& func_name){
        current_func_name = func_name;
        signature_table[func_name] = std::unordered_map<std::string, mlir::Value>();
    }

    void MLIRGenImpl::add_to_sign_table(const std::string& arg_name, mlir::Value arg){
        signature_table[current_func_name][arg_name] = arg;
    }

    mlir::Value MLIRGenImpl::get_from_sign_table(const std::string& arg_name){
        if(signature_table.count(current_func_name)){
            if (signature_table[current_func_name].count(arg_name)){
                return signature_table[current_func_name][arg_name];
            }else{
                // raise: not support yet
                mlir::emitError(mlir::UnknownLoc::get(builder.getContext()),
                    "Variable not declare: " + arg_name);
                std::exit(1);
                return nullptr;
            }
        }else{
            // raise: not support yet
            mlir::emitError(mlir::UnknownLoc::get(builder.getContext()),
                "Function not declare: " + current_func_name);
            std::exit(1);
            return nullptr;
        }
    }

    void MLIRGenImpl::add_func_to_sign_table(const std::string& func_name, mlir::func::FuncOp func){
        signature_table_func[func_name] = func;
    }

    mlir::func::FuncOp MLIRGenImpl::get_func_from_sign_table(const std::string& func_name){
        if(signature_table_func.count(func_name)){
            return signature_table_func[func_name];
        }else{
            // fail
            // raise: not support yet
            mlir::emitError(mlir::UnknownLoc::get(builder.getContext()),
                "Function not declare: " + func_name);
            std::exit(1);
            return nullptr;
        }
        
    }


    void MLIRGenImpl::parse_func(const boost::property_tree::ptree& ast){
        std::cout << "parse_func" << std::endl;

        const std::string func_name = safe_get_str(get_item(ast, 1), "text");
        init_func_in_sign_table(func_name);
        
        // Parse function args(each is mlir::Type)
        auto args = parse_func_args( safe_get_child( get_item(ast, 3), "func_param_list"));
        std::vector<mlir::Type> args_types = args.first;
        std::vector<std::string> args_names = args.second;

        // Parse function return type
        // return null for now.
        auto ret_type = builder.getNoneType();

        // Make function node
        auto func_type = builder.getFunctionType(args_types, {});
        auto func = builder.create<mlir::func::FuncOp>(loc, func_name, func_type);
        if (func_name!="main"){
            func.setPrivate();
        }
        mlir::Block *func_body = func.addEntryBlock();

        // Signature table
        for(int i=0; i<args_names.size(); i++){
            std::string name = args_names[i];
            mlir::Value arg = func_body->getArgument(i);
            add_to_sign_table(name, arg);
        }

        // Parse function body
        // std::advance(it, 3);
        // auto current_position = builder.getInsertionPoint();
        block_stack.push(func_body);
        builder.setInsertionPointToStart(func_body);

        parse_func_body(safe_get_child(get_item(ast, 6),"func_body"));

        block_stack.pop();
        builder.setInsertionPointToEnd(block_stack.top());

        // mlir::Value func_arg0 = func_body->getArgument(0);
        // mlir::Value func_arg1 = func_body->getArgument(1);
        // llvm::ArrayRef<int64_t> shape = {3,3,1,1};
        // mlir::Value a = builder.create<mlir::tensor::EmptyOp>(loc, shape, builder.getI32Type());
        // mlir::Value b = builder.create<mlir::cim::VVAddOp>(loc, func_arg0, func_arg1);
        // mlir::Value c = builder.create<mlir::cim::VVAddOp>(loc, a, func_arg1);
        
        add_func_to_sign_table(func_name, func);

        std::cout << "parse_func finish." << std::endl;
    }

    void MLIRGenImpl::parse_func_body(const boost::property_tree::ptree& ast){
        std::cout << "parse_func_body" << std::endl;

        parse_stmt_list(safe_get_child(get_item(ast,0), "stmt_list"));
        builder.create<mlir::func::ReturnOp>(loc);
        std::cout << "parse_func_body finish." << std::endl;
    }

    void MLIRGenImpl::parse_stmt_list(const boost::property_tree::ptree& ast){
        std::cout << "parse_stmt_list" << std::endl;

        for (const auto& pair : ast) {
            parse_stmt(safe_get_child(pair.second,"stmt"));
        }
    }

    void MLIRGenImpl::parse_stmt(const boost::property_tree::ptree& ast){
        std::cout << "parse_stmt" << std::endl;
        auto ast_stmt = get_item(ast, 0);
        if(is_assign_stmt(ast_stmt)){
            parse_assign_stmt(safe_get_child(ast_stmt, "stmt_assign"));
        }else if(is_return_stmt(ast_stmt)){
            // return nullptr; //parse_return_stmt(ast.begin()->first);
        }else if(is_call_stmt(ast_stmt)){
            parse_call_stmt(safe_get_child(ast_stmt, "stmt_call"));
        }else if(is_for_stmt(ast_stmt)){
            parse_for_stmt(safe_get_child(ast_stmt, "stmt_for"));
        }else {
            // raise: not support yet
            mlir::emitError(mlir::UnknownLoc::get(builder.getContext()),
                "Not support stmt: " + ast.begin()->first);
            std::exit(1);
        }
        std::cout << "parse_stmt finish" << std::endl;
    }

    bool MLIRGenImpl::is_assign_stmt(const boost::property_tree::ptree& ast){
        std::cout << "is_assign_stmt" << std::endl;

        return ast.begin()->first == "stmt_assign";
    }
    bool MLIRGenImpl::is_return_stmt(const boost::property_tree::ptree& ast){
        std::cout << "is_return_stmt" << std::endl;

        return ast.begin()->first == "stmt_return";
    }
    bool MLIRGenImpl::is_call_stmt(const boost::property_tree::ptree& ast){
        std::cout << "is_call_stmt" << std::endl;

        return ast.begin()->first == "stmt_call";
    }
    bool MLIRGenImpl::is_for_stmt(const boost::property_tree::ptree& ast){
        std::cout << "is_for_stmt" << std::endl;

        return ast.count("stmt_for");
    }
    
    /* 
        Stmt :
            stmt_assign,
            stmt_call,
            stmt_for,
            stmt_return
    
    */

    void MLIRGenImpl::parse_call_stmt(const boost::property_tree::ptree& ast){
        std::cout << "parse_call_stmt" << std::endl;
        parse_call(safe_get_child(get_item(ast, 0), "call"));
    }

    void MLIRGenImpl::parse_assign_stmt(const boost::property_tree::ptree& ast){
        std::cout << "parse_assign_stmt" << std::endl;
        // LHS
        std::string var_name = safe_get_str(get_item(ast,0), "text");

        // RHS
        mlir::Value expr = parse_expr(safe_get_child(get_item(ast,2), "expr"));
        
        // Add to sign table
        add_to_sign_table(var_name, expr);
    }

    void MLIRGenImpl::parse_for_stmt(const boost::property_tree::ptree& ast){
        std::cout << "parse_for_stmt" << std::endl;
        std::string iter_var_name = safe_get_str(get_item(ast, 1), "text");
        std::vector<mlir::Value> range = parse_for_range(safe_get_child(get_item(ast, 3), "for_range"));
        mlir::Value range_begin = range[0];
        mlir::Value range_end = range[1];
        mlir::Value range_step = range[2];

        // loop-carried variables
        auto loop_carried_names_and_variables = parse_carry(safe_get_child(get_item(ast, 4), "carry"));
        auto loop_carried_names = loop_carried_names_and_variables.first;
        auto loop_carried_variables = loop_carried_names_and_variables.second;
        mlir::scf::ForOp for_op = builder.create<mlir::scf::ForOp>(loc,range_begin, range_end, range_step, loop_carried_variables);

        // Add to sign table
        llvm::SmallVector<mlir::Value> for_args;
        for (const auto &barg : llvm::enumerate(for_op.getBody(0)->getArguments())) {
            for_args.push_back(barg.value());
        }
        add_to_sign_table(iter_var_name, for_args[0]);
        for(int i = 1;i < for_args.size();i++){ // for_args[0] is iter var
            add_to_sign_table(loop_carried_names[i-1], for_args[i]);
        }

        // Loop Body
        mlir::Block *for_body = for_op.getBody();
        block_stack.push(for_body);
        builder.setInsertionPointToStart(for_op.getBody());

        parse_stmt_list(safe_get_child(get_item(ast, 6), "stmt_list"));
        // yield
        auto yield_variables = parse_carry(safe_get_child(get_item(ast, 4), "carry")).second;
        builder.create<mlir::scf::YieldOp>(loc, yield_variables);

        block_stack.pop();
        builder.setInsertionPointToEnd(block_stack.top());

        // replace carry variables with new values
        mlir::ValueRange for_result = for_op.getResults();
        for(int i = 0;i < for_result.size();i++){ // for_args[0] is iter var
            add_to_sign_table(loop_carried_names[i], for_result[i]);
        }
    }

/*
 Stmt end
*/

/*
 * Carry begin
    carry: '(' carry_list ')';
    carry_list: var (',' var)*;
 * 
*/

std::pair<llvm::SmallVector<std::string>, llvm::SmallVector<mlir::Value>> MLIRGenImpl::parse_carry(const boost::property_tree::ptree& ast){
    std::cout << "parse_carry" << std::endl;
    auto ast_carry_list = safe_get_child(get_item(ast, 2), "carry_list");
    std::pair<llvm::SmallVector<std::string>, llvm::SmallVector<mlir::Value>> carry_list = parse_carry_list(ast_carry_list);
    std::cout << "parse_carry finish" << std::endl;
    return carry_list;
}
std::pair<llvm::SmallVector<std::string>, llvm::SmallVector<mlir::Value>> MLIRGenImpl::parse_carry_list(const boost::property_tree::ptree& ast){
    std::cout << "parse_carry_list" << std::endl;
    
    llvm::SmallVector<std::string> var_name_list;
    llvm::SmallVector<mlir::Value> vec_carry_list;
    for (const auto& pair : ast) {
        if(pair.second.count("var")){
            auto ast_var = safe_get_child(pair.second, "var");
            auto var_and_name = parse_var_and_name(ast_var);
            var_name_list.push_back(var_and_name.first);
            vec_carry_list.push_back(var_and_name.second);
        }
    }
    // mlir::ValueRange carry_list(vec_carry_list);
    std::cout << "parse_carry_list finish" << std::endl;
    return make_pair(var_name_list, vec_carry_list);
}

/*
 Range begin

    for_range : for_range_1 | for_range_2 | for_range_3;
    for_range_1: 'range(' const_or_var ')';
    for_range_2: 'range(' const_or_var ',' const_or_var')';
    for_range_3: 'range(' const_or_var ',' const_or_var ',' const_or_var ')';

    support for_range_1 first.
*/

std::vector<mlir::Value> MLIRGenImpl::parse_for_range(const boost::property_tree::ptree& ast){
    std::cout << "parse_for_range" << std::endl;

    std::vector<mlir::Value> range_values;
    auto ast_for = get_item(ast, 0);
    if(ast_for.count("for_range_1")){
        range_values = parse_for_range_1(safe_get_child(ast_for, "for_range_1"));
    }else if(ast_for.count("for_range_2")){
        range_values = parse_for_range_2(safe_get_child(ast_for, "for_range_2"));
    }else if(ast_for.count("for_range_3")){
        range_values = parse_for_range_3(safe_get_child(ast_for, "for_range_3"));
    }else{
        mlir::emitError(mlir::UnknownLoc::get(builder.getContext()),
                "Not support range");
        std::exit(1);
    }
    std::cout << "parse_range finish" << std::endl;
    return range_values;
}

std::vector<mlir::Value> MLIRGenImpl::parse_for_range_1(const boost::property_tree::ptree& ast){
    std::cout << "parse_for_range_1" << std::endl;

    mlir::Value begin = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value end = parse_const_or_var(safe_get_child(get_item(ast, 1), "const_or_var"));
    mlir::Value stride = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);

    std::cout << "parse_for_range_1 finish" << std::endl;
    return {begin, end, stride};
}

std::vector<mlir::Value> MLIRGenImpl::parse_for_range_2(const boost::property_tree::ptree& ast){
    std::cout << "parse_for_range_2" << std::endl;

    mlir::Value begin = parse_const_or_var(safe_get_child(get_item(ast, 1), "const_or_var"));
    mlir::Value end = parse_const_or_var(safe_get_child(get_item(ast, 3), "const_or_var"));
    mlir::Value stride = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);

    std::cout << "parse_for_range_2 finish" << std::endl;
    return {begin, end, stride};
}

std::vector<mlir::Value> MLIRGenImpl::parse_for_range_3(const boost::property_tree::ptree& ast){
    std::cout << "parse_for_range_3" << std::endl;

    mlir::Value begin = parse_const_or_var(safe_get_child(get_item(ast, 1), "const_or_var"));
    mlir::Value end = parse_const_or_var(safe_get_child(get_item(ast, 3), "const_or_var"));
    mlir::Value stride = parse_const_or_var(safe_get_child(get_item(ast, 5), "const_or_var"));

    std::cout << "parse_for_range_3 finish" << std::endl;
    return {begin, end, stride};
}

/*
 Range end
*/

    
mlir::Value MLIRGenImpl::parse_call_return_value(const boost::property_tree::ptree& ast){
    // call a function, and get return value

    std::cout << "parse_call_return_value" << std::endl;
    std::string call_func_name = safe_get_str(get_item(ast, 0), "text");
    
    if (call_func_name=="Shape") {
        return parse_bulitin_shape(ast);
    }else if (call_func_name=="Slice") {
        return parse_bulitin_slice(ast);
    }else if (call_func_name=="Buffer") {
        return parse_bulitin_buffer(ast);
    }else if(call_func_name=="Load") {
        return parse_bulitin_load(ast);
    }

    // check sign table
    mlir::ValueRange param_list = parse_call_param_list(safe_get_child(get_item(ast, 2), "call_param_list"));
    mlir::func::FuncOp func = get_func_from_sign_table(call_func_name);
    mlir::func::CallOp call = builder.create<mlir::func::CallOp>(loc, func, param_list);

    return call.getResult(0);
}

void MLIRGenImpl::parse_call(const boost::property_tree::ptree& ast){
    // call a function, and get return value

    std::cout << "parse_call" << std::endl;
    std::string call_func_name = safe_get_str(get_item(ast, 0), "text");
    
    if (call_func_name=="Trans") {
        parse_bulitin_trans(ast);
        return ;
    }else if (call_func_name=="VVAdd") {
        parse_bulitin_vvadd(ast);
        return ;
    }else if (call_func_name=="Print") {
        parse_bulitin_print(ast);
        return;
    }else if(call_func_name=="Free") {
        parse_bulitin_free(ast);
        return;
    }else if(call_func_name=="CIMCompute"){
        parse_bulitin_cimcompute(ast);
        return;
    }

    // check sign table
    mlir::func::FuncOp func = get_func_from_sign_table(call_func_name);

    llvm::SmallVector<mlir::Value> param_list = parse_call_param_list(safe_get_child(get_item(ast, 2), "call_param_list"));
    mlir::func::CallOp call = builder.create<mlir::func::CallOp>(loc, func, param_list);
    std::cout << "parse_call finish" << std::endl;
    return;
}

llvm::SmallVector<mlir::Value> MLIRGenImpl::parse_call_param_list(const boost::property_tree::ptree& ast){
    std::cout << "parse_call_param_list" << std::endl;
    llvm::SmallVector<mlir::Value> vec_param_list;
    for (const auto& pair : ast) {
        if(pair.second.count("call_param")){
            auto ast_call_param = safe_get_child(pair.second, "call_param");
            vec_param_list.push_back(parse_call_param(ast_call_param));
        }
    }
    // mlir::ValueRange param_list(vec_param_list);
    std::cout << "parse_call_param_list finish" << std::endl;
    return vec_param_list;
}

mlir::Value MLIRGenImpl::parse_call_param(const boost::property_tree::ptree& ast){
    std::cout << "parse_call_param" << std::endl;
    auto ast_expr = safe_get_child(get_item(ast, 0), "expr");
    return parse_expr(ast_expr);

}

/*
 * Bulitin Functions Begin
 */

mlir::Value MLIRGenImpl::parse_bulitin_shape(const boost::property_tree::ptree& ast){
    std::cout << "parse_bulitin_shape" << std::endl;
    auto ast_param_list = safe_get_child(get_item(ast,2), "call_param_list");

    auto ast_buffer = safe_get_child(get_item(ast_param_list,0), "call_param");
    auto ast_index = safe_get_child(get_item(ast_param_list,2), "call_param");

    mlir::Value buffer = parse_expr(safe_get_child(get_item(ast_buffer, 0), "expr"));
    mlir::Value index = parse_expr(safe_get_child(get_item(ast_index, 0), "expr"));
    return builder.create<mlir::cim::ShapeOp>(loc, buffer, index);
}

void MLIRGenImpl::parse_bulitin_trans(const boost::property_tree::ptree& ast){
    std::cout << "parse_bulitin_trans" << std::endl;
    auto ast_param_list = safe_get_child(get_item(ast,2), "call_param_list");
    
    auto ast_src = safe_get_child(get_item(ast_param_list,0), "call_param");
    auto ast_dst = safe_get_child(get_item(ast_param_list,2), "call_param");

    mlir::Value src = parse_expr(safe_get_child(get_item(ast_src, 0), "expr"));
    mlir::Value dst = parse_expr(safe_get_child(get_item(ast_dst, 0), "expr"));
    builder.create<mlir::cim::CopyOp>(loc, src, dst);
}

mlir::Value MLIRGenImpl::parse_bulitin_slice(const boost::property_tree::ptree& ast){
    std::cout << "parse_bulitin_slice" << std::endl;
    auto ast_param_list = safe_get_child(get_item(ast,2), "call_param_list");

    auto ast_src = safe_get_child(get_item(ast_param_list,0), "call_param");
    auto ast_offsets = safe_get_child(get_item(ast_param_list,2), "call_param");
    auto ast_sizes = safe_get_child(get_item(ast_param_list,4), "call_param");
    auto ast_strides = safe_get_child(get_item(ast_param_list,6), "call_param");
    
    mlir::Value src = parse_expr(safe_get_child(get_item(ast_src, 0), "expr"));
    mlir::SmallVector<mlir::Value> offsets = parse_array_1d(safe_get_child(get_item(ast_offsets, 0), "array1d"));
    std::cout << "parse_bulitin_slice offsets finish" << std::endl;
    std::cout << offsets.size() << std::endl;
    mlir::SmallVector<mlir::Value> sizes = parse_array_1d(safe_get_child(get_item(ast_sizes, 0), "array1d"));
    std::cout << "parse_bulitin_slice sizes finish" << std::endl;
    std::cout << sizes.size() << std::endl;
    mlir::SmallVector<mlir::Value> strides = parse_array_1d(safe_get_child(get_item(ast_strides, 0), "array1d"));
    std::cout << "parse_bulitin_slice strides finish" << std::endl;
    std::cout << strides.size() << std::endl;
    mlir::Value result = builder.create<mlir::memref::SubViewOp>(loc, src, cast_to_index_type(offsets), cast_to_index_type(sizes), cast_to_index_type(strides));
    std::cout << "parse_bulitin_slice finish" << std::endl;
    return result;
}

void MLIRGenImpl::parse_bulitin_vvadd(const boost::property_tree::ptree& ast){
    std::cout << "parse_bulitin_vvadd"  << std::endl;
    auto ast_param_list = safe_get_child(get_item(ast,2), "call_param_list");

    auto ast_lhs = safe_get_child(get_item(ast_param_list,0), "call_param");
    auto ast_rhs = safe_get_child(get_item(ast_param_list,2), "call_param");
    auto ast_out = safe_get_child(get_item(ast_param_list,4), "call_param");

    mlir::Value lhs = parse_expr(safe_get_child(get_item(ast_lhs, 0), "expr"));
    mlir::Value rhs = parse_expr(safe_get_child(get_item(ast_rhs, 0), "expr"));
    mlir::Value out = parse_expr(safe_get_child(get_item(ast_out, 0), "expr"));
    builder.create<mlir::cim::VVAddOp>(loc, lhs, rhs, out);
}

mlir::Value MLIRGenImpl::parse_bulitin_buffer(const boost::property_tree::ptree& ast){
    std::cout << "parse_bulitin_buffer" << std::endl;
    auto ast_param_list = safe_get_child(get_item(ast,2), "call_param_list");

    // Data type
    auto ast_dtype_call_param = safe_get_child(get_item(ast_param_list, 2), "call_param");
    auto ast_dtype = safe_get_child(get_item(ast_dtype_call_param, 0), "datatype");
    std::string str_dtype = safe_get_str(get_item(ast_dtype,0), "text");
    mlir::Type dtype = parse_datatype(str_dtype);

    // Memory type
    auto ast_memory_call_param = safe_get_child(get_item(ast_param_list, 4), "call_param");
    auto ast_memory = safe_get_child(get_item(ast_memory_call_param, 0), "memory");
    std::string memory = safe_get_str(get_item(ast_memory,0), "text");
    mlir::Attribute memory_attr = parse_device(memory);

    // Shape
    auto ast_shape = safe_get_child(get_item(ast_param_list, 0), "call_param");
    auto ast_shape_array1d = safe_get_child(get_item(ast_shape,0), "const_array1d");
    std::vector<int64_t> shape = parse_const_array1d(ast_shape_array1d);
    
    mlir::MemRefType type =  mlir::MemRefType::get(
        llvm::ArrayRef<int64_t>(shape), 
        dtype, 
        mlir::MemRefLayoutAttrInterface(), 
        memory_attr
    );
    mlir::memref::AllocOp alloc = builder.create<mlir::memref::AllocOp>(loc, type);
    return alloc.getResult();
}

void MLIRGenImpl::parse_bulitin_print(const boost::property_tree::ptree& ast){
    std::cout << "parse_bulitin_print" << std::endl;
    auto ast_param_list = safe_get_child(get_item(ast,2), "call_param_list");

    auto ast_value = safe_get_child(get_item(ast_param_list,0), "call_param");

    mlir::Value value = parse_expr(safe_get_child(get_item(ast_value, 0), "expr"));
    builder.create<mlir::cim::PrintOp>(loc, value);
}

void MLIRGenImpl::parse_bulitin_free(const boost::property_tree::ptree& ast){
    std::cout << "parse_bulitin_free" << std::endl;
    auto ast_param_list = safe_get_child(get_item(ast,2), "call_param_list");

    auto ast_value = safe_get_child(get_item(ast_param_list,0), "call_param");

    mlir::Value value = parse_expr(safe_get_child(get_item(ast_value, 0), "expr"));
    builder.create<mlir::memref::DeallocOp>(loc, value);
}

mlir::Value MLIRGenImpl::parse_bulitin_load(const boost::property_tree::ptree& ast){
    std::cout << "parse_bulitin_load" << std::endl;
    auto ast_param_list = safe_get_child(get_item(ast,2), "call_param_list");

    auto ast_memref = safe_get_child(get_item(ast_param_list,0), "call_param");
    auto ast_index = safe_get_child(get_item(ast_param_list,2), "call_param");

    mlir::Value memref = parse_expr(safe_get_child(get_item(ast_memref, 0), "expr"));
    mlir::SmallVector<mlir::Value> _index = parse_array_1d(safe_get_child(get_item(ast_index, 0), "array1d"));
    mlir::SmallVector<mlir::Value> index = cast_to_index_type(_index);

    mlir::Value result = builder.create<mlir::memref::LoadOp>(loc, memref, index);
    return result;
}

void MLIRGenImpl::parse_bulitin_cimcompute(const boost::property_tree::ptree& ast){
    std::cout << "parse_bulitin_cimcompute" << std::endl;
    auto ast_param_list = safe_get_child(get_item(ast,2), "call_param_list");

    auto ast_input = safe_get_child(get_item(ast_param_list,0), "call_param");
    auto ast_macro = safe_get_child(get_item(ast_param_list,2), "call_param");
    auto ast_output = safe_get_child(get_item(ast_param_list,4), "call_param");

    mlir::Value input = parse_expr(safe_get_child(get_item(ast_input, 0), "expr"));
    mlir::Value macro = parse_expr(safe_get_child(get_item(ast_macro, 0), "expr"));
    mlir::Value output = parse_expr(safe_get_child(get_item(ast_output, 0), "expr"));
    builder.create<mlir::cim::CIMComputeOp>(loc, input, macro, output);
}

/*
 * Bulitin Functions End
 */
 
    // mlir::Value parse_var(const boost::property_tree::ptree& ast){
    //     auto constant = get_item(ast, 0);
    //     int value = std::stoi(constant.get<std::string>("text"));
    // }

    mlir::Value MLIRGenImpl::parse_var(const boost::property_tree::ptree& ast){
        std::cout << "parse_var" << std::endl;
        std::string var_name = safe_get_str(get_item(ast, 0), "text");
        mlir::Value var = get_from_sign_table(var_name);
        return var;
    }

    std::pair<std::string, mlir::Value> MLIRGenImpl::parse_var_and_name(const boost::property_tree::ptree& ast){
        std::cout << "parse_var" << std::endl;
        std::string var_name = safe_get_str(get_item(ast, 0), "text");
        mlir::Value var = get_from_sign_table(var_name);
        return std::make_pair(var_name, var);
    }

    int64_t MLIRGenImpl::parse_const_int(const boost::property_tree::ptree& ast){
        std::cout << "parse_const_int" << std::endl;

        auto const_node = get_item(ast, 0);
        int value = std::stoi(safe_get_str(const_node, "text"));
        return static_cast<int64_t>(value);
    }

    mlir::Value MLIRGenImpl::parse_const(const boost::property_tree::ptree& ast){
        std::cout << "parse_const" << std::endl;

        auto const_node = get_item(ast, 0);
        int value = std::stoi(safe_get_str(const_node, "text"));
        mlir::Value const_value = builder.create<mlir::arith::ConstantIndexOp>(loc, value);
        std::cout << "parse_const finish" << std::endl;
        return const_value;
    }

    bool MLIRGenImpl::is_const(const boost::property_tree::ptree& ast){
        std::cout << "is_const" << std::endl;
        return ast.count("constant");
    }

    bool MLIRGenImpl::is_var(const boost::property_tree::ptree& ast){
        std::cout << "is_var" << std::endl;
        return ast.count("var");
    }

    mlir::Value MLIRGenImpl::parse_const_or_var(const boost::property_tree::ptree& ast){
        std::cout <<    "parse_const_or_var" << std::endl;

        auto const_or_var = get_item(ast, 0);
        if(is_const(const_or_var)){
            return parse_const(safe_get_child(const_or_var, "constant"));
        }else if(is_var(const_or_var)){
            return parse_var( safe_get_child( const_or_var, "var"));
        }
    }

    bool MLIRGenImpl::is_const_or_var(const boost::property_tree::ptree& ast){
        std::cout << "is_const_or_var" << std::endl;
        return ast.count("const_or_var");
    }

    bool MLIRGenImpl::is_call(const boost::property_tree::ptree& ast){
        std::cout << "is_call" << std::endl;
        return ast.count("call");
    }

    mlir::Value MLIRGenImpl::parse_unary_expr(const boost::property_tree::ptree& ast){
        /*
            unary_expr: call | const_or_var;
        */
        std::cout << "parse_unary_expr" << std::endl;
        auto unary_expr = get_item(ast, 0);
        if (is_const_or_var(unary_expr)){
            return parse_const_or_var( safe_get_child( unary_expr, "const_or_var"));
        }else if(is_call(unary_expr)){
            return parse_call_return_value( safe_get_child( unary_expr, "call"));
        }else{
            // raise: not support yet
            mlir::emitError(mlir::UnknownLoc::get(builder.getContext()),
                "Not support unary_expr: " + ast.begin()->first);
            std::exit(1);
            return nullptr;
        }
    }

    mlir::Value MLIRGenImpl::parse_binary_expr(const boost::property_tree::ptree& ast){
        /*
            binary_expr: unary_expr BINARY_OP unary_expr;
        */
        std::cout << "parse_binary_expr" << std::endl;
        auto ast_lhs =  safe_get_child( get_item(ast, 0), "unary_expr");
        mlir::Value lhs = parse_unary_expr(ast_lhs);

        auto binary_op = safe_get_str(get_item(ast, 1), "text");
        
        auto ast_rhs =  safe_get_child( get_item(ast, 2), "unary_expr");
        mlir::Value rhs = parse_unary_expr(ast_rhs);

        if (binary_op == "+"){
            return builder.create<mlir::arith::AddIOp>(loc, lhs, rhs);
        }else if (binary_op == "-"){
            return builder.create<mlir::arith::SubIOp>(loc, lhs, rhs);
        }else if (binary_op == "*"){
            return builder.create<mlir::arith::MulIOp>(loc, lhs, rhs);
        }else if (binary_op == "/"){
            return builder.create<mlir::arith::DivSIOp>(loc, lhs, rhs);
        }else if(binary_op == "%"){
            return builder.create<mlir::arith::RemSIOp>(loc, lhs, rhs);
        }else{
            // raise: not support yet
            mlir::emitError(mlir::UnknownLoc::get(builder.getContext()),
                "Not support binary_op: " + binary_op);
            std::exit(1);
            return nullptr;
        }
        
    }

    bool MLIRGenImpl::is_unary_expr(const boost::property_tree::ptree& ast){
        std::cout << "is_unary_expr" << std::endl;
        return ast.count("unary_expr");
    }

    bool MLIRGenImpl::is_binary_expr(const boost::property_tree::ptree& ast){
        std::cout << "is_binary_expr" << std::endl;
        return ast.count("binary_expr");
    }

    mlir::Value MLIRGenImpl::parse_expr(const boost::property_tree::ptree& ast){
        std::cout <<    "parse_expr" << std::endl;
        auto expr = get_item(ast, 0);
        if(is_unary_expr(expr)){
            return parse_unary_expr( safe_get_child( expr, "unary_expr"));
        }else if(is_binary_expr(expr)){
            return parse_binary_expr( safe_get_child( expr, "binary_expr"));
        }else{
            // raise: not support yet
            mlir::emitError(mlir::UnknownLoc::get(builder.getContext()),
                "Not support expr: " + ast.begin()->first);
            std::exit(1);
            return nullptr;
        }
    }

    mlir::SmallVector<mlir::Value> MLIRGenImpl::parse_array_1d(const boost::property_tree::ptree& ast){
        std::cout << "parse_array_1d" << std::endl;
        mlir::SmallVector<mlir::Value> values;
        for (const auto& pair : ast) {
            if(pair.second.count("text") && (
                safe_get_str(pair.second, "text")=="[" or 
                safe_get_str(pair.second, "text")=="]" or
                safe_get_str(pair.second, "text")=="," )){
                continue;
            }
            values.push_back(parse_expr( safe_get_child( pair.second, "expr")));
        }
        std::cout << "parse_array_1d finish" << std::endl;
        return values;
    }

    std::vector<int64_t> MLIRGenImpl::parse_const_array1d(const boost::property_tree::ptree& ast){
        std::cout << "parse_const_array1d" << std::endl;

        std::vector<int64_t> values;
        for (const auto& pair : ast) {
            if(pair.second.count("text") && (
                safe_get_str(pair.second, "text")=="<" or 
                safe_get_str(pair.second, "text")==">" or
                safe_get_str(pair.second, "text")=="," )){
                continue;
            }
            values.push_back(parse_const_int( safe_get_child( pair.second, "constant")));
        }
        return values;
         
    }

    std::vector<int64_t> MLIRGenImpl::parse_shape(const boost::property_tree::ptree& ast){
        std::cout << "parse_shape" << std::endl;

        std::vector<int64_t> shape = parse_const_array1d( safe_get_child( get_item(ast, 0), "const_array1d"));
        std::vector<int64_t> adjust_shape;
        for(auto it=shape.begin();it!=shape.end();it++){
            int64_t value = *it;
            if(value > 0){
                adjust_shape.push_back(value);
            }else{
                adjust_shape.push_back(mlir::ShapedType::kDynamic);
            }
        }
        return adjust_shape;
        // return llvm::makeArrayRef(shape.begin(), shape.size());
    }

    mlir::Type MLIRGenImpl::parse_datatype(std::string datatype){
        std::cout << "parse_datatype" << std::endl;

        if(datatype=="int8"){
            return builder.getI8Type();
        }else if(datatype=="int32"){
            return builder.getI32Type();
        }else if(datatype=="int64"){
            return builder.getI64Type();
        }else if(datatype=="index"){
            return builder.getIndexType();
        }else if(datatype=="float32"){
            return builder.getF32Type();
        }else{
            // raise: not support yet
            mlir::emitError(mlir::UnknownLoc::get(builder.getContext()),
                "Not support datatype: " + datatype);
            std::exit(1);
            return nullptr;
        }
    }

    mlir::Attribute MLIRGenImpl::parse_device(std::string device){
        std::cout << "parse_device" << std::endl;

        if (device=="global" || device=="local" || device=="macro"|| device=="rf"){
            mlir::SmallVector<mlir::NamedAttribute, 2> nameAttrs;
            nameAttrs.push_back(builder.getNamedAttr("memory", builder.getStringAttr(device)));
            nameAttrs.push_back(builder.getNamedAttr("address", builder.getI64IntegerAttr(-1)));
            mlir::DictionaryAttr attr = mlir::DictionaryAttr::get(builder.getContext(), nameAttrs);
            return attr;
        }else{
            // raise: not support yet
            mlir::emitError(mlir::UnknownLoc::get(builder.getContext()),
                "Not support device: " + device);
            return nullptr;
        }
    }

    mlir::MemRefType MLIRGenImpl::parse_param_type_tensor(const boost::property_tree::ptree& ast) {
        std::cout << "parse_param_type_tensor" << std::endl;

        // shape
        auto param_type_shape =  safe_get_child( get_item(ast, 1), "param_type_shape");
        auto shape = parse_shape(param_type_shape);

        // datatype
        auto datatype = parse_datatype(safe_get_str(get_item(ast, 3), "text"));

        // device
        auto device = parse_device(safe_get_str(get_item(ast, 5), "text"));

        // build the type
        mlir::MemRefType type =  mlir::MemRefType::get(
            llvm::ArrayRef<int64_t>(shape), 
            datatype, 
            mlir::MemRefLayoutAttrInterface(), 
            device
        );
        return type;
    }

    mlir::Type MLIRGenImpl::parse_param_type_scalar(const boost::property_tree::ptree& ast) {
        std::cout << "parse_param_type_scalar" << std::endl;

        // datatype
        auto datatype = parse_datatype(safe_get_str(get_item(ast, 1), "text"));

        return datatype;
    }

    mlir::Type MLIRGenImpl::parse_param_type(const boost::property_tree::ptree& ast) {
        std::cout << "parse_param_type" << std::endl;
        auto ast_param_type = get_item(ast, 0);
        if (ast_param_type.count("param_type_tensor")){
            return parse_param_type_tensor( safe_get_child( ast_param_type, "param_type_tensor"));
        }else if (ast_param_type.count("param_type_scalar")){
            return parse_param_type_scalar( safe_get_child( ast_param_type, "param_type_scalar"));
        }else{
            // raise: not support yet
            mlir::emitError(mlir::UnknownLoc::get(builder.getContext()),
                "Not support param_type: " + ast_param_type.begin()->first);
            std::exit(1);
            return nullptr;
        }
    }

    std::pair<mlir::Type, std::string> MLIRGenImpl::parse_param_type_and_name(const boost::property_tree::ptree& ast){
        std::cout << "parse_tensor_type_and_name" << std::endl;
        
        auto args_name =  safe_get_child( get_item(ast, 0), "param_name");
        std::string name = safe_get_str(get_item(args_name, 0), "text");

        mlir::Type args_type = parse_param_type( safe_get_child( get_item(ast, 1), "param_type"));
        return std::make_pair(args_type, name);
    }

    std::pair<std::vector<mlir::Type>, std::vector<std::string> > MLIRGenImpl::parse_func_args(const boost::property_tree::ptree& ast){
        std::cout << "parse_func_args" << std::endl;

        std::vector<mlir::Type> args_types;
        std::vector<std::string> args_names;
        int i = 0;
        for (const auto& pair : ast) {
            if(is_tensor_args(pair.second) || is_scalar_args(pair.second)){
                auto type_and_name = parse_param_type_and_name( safe_get_child( pair.second, "func_param"));
                args_types.push_back(type_and_name.first);
                args_names.push_back(type_and_name.second);
            }
        }
        return std::make_pair(args_types, args_names);
    }

    bool MLIRGenImpl::is_tensor_args(const boost::property_tree::ptree& ast){
        std::cout << "is_tensor_args" << std::endl;
        
        if(!ast.count("func_param")) return false;
        auto func_param =  safe_get_child( ast, "func_param");
        auto param_type =  safe_get_child( get_item(func_param, 1), "param_type");
        return get_item(param_type, 0).count("param_type_tensor");
    }

    bool MLIRGenImpl::is_scalar_args(const boost::property_tree::ptree& ast){
        std::cout << "is_scalar_args" << std::endl;

        if(!ast.count("func_param")) return false;
        auto func_param =  safe_get_child( ast, "func_param");
        auto param_type =  safe_get_child( get_item(func_param, 1), "param_type");
        return get_item(param_type, 0).count("param_type_scalar");
    }

    mlir::SmallVector<mlir::Value> MLIRGenImpl::cast_to_index_type(mlir::SmallVector<mlir::Value> _index){
        // cast to index
        // mlir::SmallVector<mlir::Value> index;
        // for(int i = 0; i<_index.size(); i++){
        //     mlir::Value index_i = builder.create<mlir::index::CastSOp>(loc, builder.getIndexType(), _index[i]);
        //     index.push_back(index_i);
        // }
        return _index;
    }