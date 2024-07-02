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
        builder.setInsertionPointToEnd(module.getBody());
        
        auto program = safe_get_child(ast, "program");
        auto define_func_list = get_item(program, 0);

        for (const auto& pair : program) {
            auto ast_define_func = safe_get_child(pair.second, "define_function");
            parse_func(ast_define_func);
        }
        return module;
    }

    void MLIRGenImpl::init_func_in_sign_table(const std::string& func_name){
        current_func_name = func_name;
        signature_table[func_name] = std::unordered_map<std::string, mlir::Value>();
    }

    void MLIRGenImpl::add_to_sign_table(const std::string& arg_name, mlir::Value& arg){
        signature_table[current_func_name][arg_name] = arg;
    }

    void MLIRGenImpl::add_func_to_sign_table(const std::string& func_name, mlir::func::FuncOp& func){
        signature_table_func[func_name] = func;
    }

    mlir::func::FuncOp MLIRGenImpl::get_func_from_sign_table(const std::string& func_name){
        return signature_table_func[func_name];
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
        auto func_type = builder.getFunctionType(args_types, {ret_type});
        auto func = builder.create<mlir::func::FuncOp>(loc, func_name, func_type);
        func.setPrivate();
        mlir::Block *func_body = func.addEntryBlock();

        // Signature table
        for(int i=0; i<args_names.size(); i++){
            std::string name = args_names[i];
            mlir::Value arg = func_body->getArgument(i);
            add_to_sign_table(name, arg);
        }

        // Parse function body
        // std::advance(it, 3);
        // builder.setInsertionPointToStart(func_body);
        parse_func_body(safe_get_child(get_item(ast, 6),"func_body"));
        // mlir::Value func_arg0 = func_body->getArgument(0);
        // mlir::Value func_arg1 = func_body->getArgument(1);
        // llvm::ArrayRef<int64_t> shape = {3,3,1,1};
        // mlir::Value a = builder.create<mlir::tensor::EmptyOp>(loc, shape, builder.getI32Type());
        // mlir::Value b = builder.create<mlir::cim::VVAddOp>(loc, func_arg0, func_arg1);
        // mlir::Value c = builder.create<mlir::cim::VVAddOp>(loc, a, func_arg1);

        add_func_to_sign_table(func_name, func);
    }

    void MLIRGenImpl::parse_func_body(const boost::property_tree::ptree& ast){
        std::cout << "parse_func_body" << std::endl;

        parse_stmt_list(safe_get_child(get_item(ast,0), "stmt_list"));
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
        }else {
            // raise: not support yet
            mlir::emitError(mlir::UnknownLoc::get(builder.getContext()),
                "Not support stmt: " + ast.begin()->first);
            std::exit(1);
        }
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

        return ast.begin()->first == "stmt_return";
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
        // LHS
        std::string var_name = safe_get_str(get_item(ast,0), "text");

        // RHS
        mlir::Value expr = parse_expr(safe_get_child(get_item(ast,2), "expr"));
        
        // Add to sign table
        add_to_sign_table(var_name, expr);
    }
    

    // mlir::Value MLIRGenImpl::parse_expr(const boost::property_tree::ptree& ast){
    //     /*
    //         ast: 
    //             {
    //                 "unary_expr": [
    //                     ....
    //                 ]
    //             }
    //     */
    //     std::cout << "parse_expr" << std::endl;

    //     if (is_unary_expr(ast)){
    //         return parse_unary_expr(ast.begin()->second.begin()->second);
    //     }else if(is_binary_expr(ast)){
    //         return nullptr;// parse_binary_expr(ast.begin()->second);
    //     }else{
    //         return nullptr;
    //     }
    // }

    // bool MLIRGenImpl::is_unary_expr(const boost::property_tree::ptree& ast){
    //     /*
    //         ast: 
    //             {
    //                 "expr_unary": [
    //                     ....
    //                 ]
    //             }
    //     */
    //     std::cout << "is_unary_expr" << std::endl;

    //     return ast.begin()->first == "expr_unary";
    // }

    // bool MLIRGenImpl::is_binary_expr(const boost::property_tree::ptree& ast){
    //     /*
    //         ast: 
    //             {
    //                 "expr_binary": [
    //                     ....
    //                 ]
    //             }
    //     */
    //     std::cout << "is_binary_expr" << std::endl;

    //     return ast.begin()->first == "expr_binary";
    // }

    // mlir::Value MLIRGenImpl::parse_unary_expr(const boost::property_tree::ptree& ast){
    //     /*
    //         ast: 
    //             {
    //                 "call": [
    //                     ...
    //                 ]
    //             }
    //     */
    //     std::cout << "parse_unary_expr" << std::endl;
    //     auto it = ast.begin();
    //     if(it->first == "call"){
    //         return parse_call_return_value(it->second);
    //     }else{
    //         return nullptr;
    //     }
    // }

mlir::Value MLIRGenImpl::parse_call_return_value(const boost::property_tree::ptree& ast){
    // call a function, and get return value

    std::cout << "parse_call_return_value" << std::endl;
    std::string call_func_name = safe_get_str(get_item(ast, 0), "text");
    mlir::ValueRange param_list = parse_call_args(safe_get_child(get_item(ast, 2), "call_param_list"));

    if (call_func_name=="Shape") {
        return parse_bulitin_shape(param_list);
    }else if (call_func_name=="Slice") {
        return parse_bulitin_slice(ast);
    }else if (call_func_name=="Buffer") {
        return parse_bulitin_buffer(ast);
    }

    // check sign table
    mlir::func::FuncOp func = get_func_from_sign_table(call_func_name);
    mlir::func::CallOp call = builder.create<mlir::func::CallOp>(loc, func, param_list);

    return call.getResult(0);
}

void MLIRGenImpl::parse_call(const boost::property_tree::ptree& ast){
    // call a function, and get return value

    std::cout << "parse_call" << std::endl;
    std::string call_func_name = safe_get_str(get_item(ast, 0), "text");
    mlir::ValueRange param_list = parse_call_args(safe_get_child(get_item(ast, 2), "call_param_list"));

    if (call_func_name=="Trans") {
        parse_bulitin_trans(param_list);
        return ;
    }else if (call_func_name=="VVAdd") {
        parse_bulitin_vvadd(param_list);
        return ;
    }

    // check sign table
    mlir::func::FuncOp func = get_func_from_sign_table(call_func_name);
    mlir::func::CallOp call = builder.create<mlir::func::CallOp>(loc, func, param_list);

    return;
}

mlir::ValueRange MLIRGenImpl::parse_call_args(const boost::property_tree::ptree& ast){
    std::cout << "parse_call_args" << std::endl;
    std::vector<mlir::Value> vec_param_list;
    for (const auto& pair : ast) {
        vec_param_list.push_back(parse_expr(pair.second));
    }
    mlir::ValueRange param_list(vec_param_list);
    return param_list;
}

/*
 * Bulitin Functions Begin
 */

mlir::Value MLIRGenImpl::parse_bulitin_shape(mlir::ValueRange param_list){
    if(param_list.size() != 1){
        std::cout << "ShapeOp only accept one parameter" << std::endl;
        std::exit(1);
        return nullptr;
    }
    mlir::Value buffer = param_list[0];
    mlir::Value index = param_list[1];
    return builder.create<mlir::cim::ShapeOp>(loc, buffer, index);
}

void MLIRGenImpl::parse_bulitin_trans(mlir::ValueRange param_list){
    mlir::Value src = param_list[0];
    mlir::Value dst = param_list[1];
    builder.create<mlir::memref::CopyOp>(loc, src, dst);
}

mlir::Value MLIRGenImpl::parse_bulitin_slice(const boost::property_tree::ptree& ast){
    auto ast_param_list = safe_get_child(get_item(ast,2), "call_param_list");

    auto ast_src = safe_get_child(get_item(ast_param_list,0), "call_param");
    auto ast_offsets = safe_get_child(get_item(ast_param_list,2), "call_param");
    auto ast_sizes = safe_get_child(get_item(ast_param_list,4), "call_param");
    
    mlir::Value src = parse_expr(safe_get_child(get_item(ast_src, 0), "expr"));
    mlir::SmallVector<mlir::Value> offsets = parse_array_1d(safe_get_child(get_item(ast_offsets, 0), "array1d"));
    mlir::SmallVector<mlir::Value> sizes = parse_array_1d(safe_get_child(get_item(ast_sizes, 0), "array1d"));
    
    mlir::Value result = builder.create<mlir::memref::SubViewOp>(loc, src, offsets, sizes, sizes);
    return result;
}

void MLIRGenImpl::parse_bulitin_vvadd(mlir::ValueRange param_list){
    mlir::Value lhs = param_list[0];
    mlir::Value rhs = param_list[1];
    builder.create<mlir::cim::VVAddOp>(loc, lhs, rhs);
}

mlir::Value MLIRGenImpl::parse_bulitin_buffer(const boost::property_tree::ptree& ast){
    auto ast_param_list = safe_get_child(get_item(ast,2), "call_param_list");

    // Data type
    auto ast_dtype = safe_get_child(get_item(ast_param_list, 2), "call_param");
    std::string str_dtype = safe_get_str(ast_dtype, "text");
    mlir::Type dtype = parse_datatype(str_dtype);

    // Memory type
    auto ast_memory = safe_get_child(get_item(ast_param_list, 4), "call_param");
    std::string memory = safe_get_str(ast_memory, "text");
    mlir::StringAttr memory_attr = parse_device(memory);

    // Shape
    auto ast_shape = safe_get_child(get_item(ast_param_list, 0), "call_param");
    auto ast_shape_array1d = safe_get_child(ast_shape, "array1d");
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

/*
 * Bulitin Functions End
 */
 
    // mlir::Value parse_var(const boost::property_tree::ptree& ast){
    //     auto constant = get_item(ast, 0);
    //     int value = std::stoi(constant.get<std::string>("text"));
    // }

    mlir::Value MLIRGenImpl::parse_var(const boost::property_tree::ptree& ast){
        std::cout << "parse_var" << std::endl;
        return nullptr;
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
        mlir::Value const_value = builder.create<mlir::arith::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(value));
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
        }
    }

    mlir::Value MLIRGenImpl::parse_binary_expr(const boost::property_tree::ptree& ast){
        /*
            binary_expr: unary_expr BINARY_OP unary_expr;
        */
        std::cout << "parse_binary_expr" << std::endl;
        auto lhs_unary_expr =  safe_get_child( get_item(ast, 0), "unary_expr");
        auto binary_op = get_item(ast, 1);
        auto rhs_unary_expr =  safe_get_child( get_item(ast, 2), "unary_expr");
        return nullptr;
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
        return values;
    }

    std::vector<int64_t> MLIRGenImpl::parse_const_array1d(const boost::property_tree::ptree& ast){
        std::cout << "parse_const_array1d" << std::endl;

        std::vector<int64_t> values;
        for (const auto& pair : ast) {
            if(pair.second.count("text") && (
                safe_get_str(pair.second, "text")=="[" or 
                safe_get_str(pair.second, "text")=="]" or
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
            return builder.getI16Type();
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

    mlir::StringAttr MLIRGenImpl::parse_device(std::string device){
        std::cout << "parse_device" << std::endl;

        if (device=="global"){
            return GLOBAL_MEMORY;
        }else if(device=="local"){
            return LOCAL_MEMORY;
        }else if(device=="macro"){
            return MACRO_MEMORY;
        }else{
            // raise: not support yet
            mlir::emitError(mlir::UnknownLoc::get(builder.getContext()),
                "Not support device: " + device);
            return nullptr;
        }
    }

    mlir::RankedTensorType MLIRGenImpl::parse_param_type_tensor(const boost::property_tree::ptree& ast) {
        std::cout << "parse_param_type_tensor" << std::endl;

        // shape
        auto param_type_shape =  safe_get_child( get_item(ast, 1), "param_type_shape");
        auto shape = parse_shape(param_type_shape);

        // datatype
        auto datatype = parse_datatype(safe_get_str(get_item(ast, 3), "text"));

        // device
        auto device = parse_device(safe_get_str(get_item(ast, 5), "text"));

        // build the type
        mlir::RankedTensorType::Builder _type_builder = 
                mlir::RankedTensorType::Builder(llvm::ArrayRef<int64_t>(shape), datatype, device); 
        // mlir::RankedTensorType::Builder _type_builder = 
        //         mlir::RankedTensorType::Builder({10,10}, builder.getI32Type(), builder.getStringAttr("global")); 
        mlir::RankedTensorType type = mlir::RankedTensorType(_type_builder);
        return type;
    }

    mlir::Type MLIRGenImpl::parse_param_type(const boost::property_tree::ptree& ast) {
        std::cout << "parse_param_type" << std::endl;
        
        return parse_param_type_tensor( safe_get_child( get_item(ast, 0), "param_type_tensor"));
    }

    std::pair<mlir::Type, std::string> MLIRGenImpl::parse_tensor_type_and_name(const boost::property_tree::ptree& ast){
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
            if(is_tensor_args(pair.second)){
                auto type_and_name = parse_tensor_type_and_name( safe_get_child( pair.second, "func_param"));
                args_types.push_back(type_and_name.first);
                args_names.push_back(type_and_name.second);
            }else if(is_scalar_args(pair.second)){
                // auto type_and_name = parse_scalar_type_and_name(pair.second.get_child("func_param"));
                // args_types.push_back(type_and_name.first);
                // args_names.push_back(type_and_name.second);
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
        return get_item(param_type, 0).count("param_scalar_tensor");
    }