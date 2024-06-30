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
        // parse_func_body(it->second.get_child("func_body"));
        mlir::Value func_arg0 = func_body->getArgument(0);
        mlir::Value func_arg1 = func_body->getArgument(1);
        llvm::ArrayRef<int64_t> shape = {3,3,1,1};
        mlir::Value a = builder.create<mlir::tensor::EmptyOp>(loc, shape, builder.getI32Type());
        mlir::Value b = builder.create<mlir::cim::VVAddOp>(loc, func_arg0, func_arg1);
        mlir::Value c = builder.create<mlir::cim::VVAddOp>(loc, a, func_arg1);

        add_func_to_sign_table(func_name, func);
    }

    void MLIRGenImpl::parse_func_body(const boost::property_tree::ptree& ast){
        std::cout << "parse_func_body" << std::endl;

        parse_stmt_list(ast.begin()->second);
    }

    void MLIRGenImpl::parse_stmt_list(const boost::property_tree::ptree& ast){
        std::cout << "parse_stmt_list" << std::endl;

        for (const auto& pair : ast) {
            parse_stmt(pair.second);
        }
    }

    void MLIRGenImpl::parse_stmt(const boost::property_tree::ptree& ast){
        std::cout << "parse_stmt" << std::endl;

        if(is_assign_stmt(ast)){
            parse_assign_stmt(ast.begin()->second);
        }else if(is_return_stmt(ast)){
            // return nullptr; //parse_return_stmt(ast.begin()->first);
        }else if(is_call_stmt(ast)){
            parse_call_stmt(get_item(ast, 0));
        }else {
            // raise: not support yet
            mlir::emitError(mlir::UnknownLoc::get(builder.getContext()),
                "Not support stmt: " + ast.begin()->first);
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
        parse_call(get_item(ast, 0));
    }

    void MLIRGenImpl::parse_assign_stmt(const boost::property_tree::ptree& ast){
        // auto it = ast.begin();
        // // LHS
        // std::string var_name = it->second.get<std::string>("text");

        // // RHS
        // std::advance(it, 2);
        // mlir::Value expr = parse_expr(it->second.get_child("expr").begin()->second);
        
        // // Add to sign table
        // add_to_sign_table(var_name, expr);
    }
    

    mlir::Value MLIRGenImpl::parse_expr(const boost::property_tree::ptree& ast){
        /*
            ast: 
                {
                    "unary_expr": [
                        ....
                    ]
                }
        */
        std::cout << "parse_expr" << std::endl;

        if (is_unary_expr(ast)){
            return parse_unary_expr(ast.begin()->second.begin()->second);
        }else if(is_binary_expr(ast)){
            return nullptr;// parse_binary_expr(ast.begin()->second);
        }else{
            return nullptr;
        }
    }

    bool MLIRGenImpl::is_unary_expr(const boost::property_tree::ptree& ast){
        /*
            ast: 
                {
                    "expr_unary": [
                        ....
                    ]
                }
        */
        std::cout << "is_unary_expr" << std::endl;

        return ast.begin()->first == "expr_unary";
    }

    bool MLIRGenImpl::is_binary_expr(const boost::property_tree::ptree& ast){
        /*
            ast: 
                {
                    "expr_binary": [
                        ....
                    ]
                }
        */
        std::cout << "is_binary_expr" << std::endl;

        return ast.begin()->first == "expr_binary";
    }

    mlir::Value MLIRGenImpl::parse_unary_expr(const boost::property_tree::ptree& ast){
        /*
            ast: 
                {
                    "call": [
                        ...
                    ]
                }
        */
        std::cout << "parse_unary_expr" << std::endl;
        auto it = ast.begin();
        if(it->first == "call"){
            return parse_call(it->second);
        }else{
            return nullptr;
        }
    }

    mlir::Value MLIRGenImpl::parse_call(const boost::property_tree::ptree& ast){
        std::cout << "parse_call" << std::endl;
        std::string call_func_name = safe_get_str(get_item(ast, 0), "text");
        mlir::ValueRange param_list = parse_param_lisr(safe_get_child(get_item(ast, 2), "call_param_list"));

        if (call_func_name=="Shape") {

        }else if (call_func_name=="Trans") {
            
        }else if ()

       
        
        builder.create<func::CallOp>(loc, call_func_name, param_list);


        return nullptr;
        // auto it = ast.begin();
        // std::string func_name = it->second.get<std::string>("text");
        // std::advance(it, 2);
        // std::vector<mlir::Value> args = parse_call_param_list(it->second.get_child("call_param_list"));
        // auto func = get_func_from_sign_table(func_name);
        // return builder.create<mlir::func::CallOp>(loc, func, args).getResults();
    }

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

    mlir::Value MLIRGenImpl::parse_unary_expr_scalar(const boost::property_tree::ptree& ast){
        /*
            unary_expr_scalar: call | const_or_var;
        */
        std::cout << "parse_unary_expr_scalar" << std::endl;
        auto unary_expr_scalar = get_item(ast, 0);
        if (is_const_or_var(unary_expr_scalar)){
            return parse_const_or_var( safe_get_child( unary_expr_scalar, "const_or_var"));
        }else if(is_call(unary_expr_scalar)){
            return parse_call( safe_get_child( unary_expr_scalar, "call"));
        }
    }

    mlir::Value MLIRGenImpl::parse_binary_expr_scalar(const boost::property_tree::ptree& ast){
        /*
            binary_expr_scalar: unary_expr_scalar BINARY_OP unary_expr_scalar;
        */
        std::cout << "parse_binary_expr_scalar" << std::endl;
        auto lhs_unary_expr_scalar =  safe_get_child( get_item(ast, 0), "unary_expr_scalar");
        auto binary_op = get_item(ast, 1);
        auto rhs_unary_expr_scalar =  safe_get_child( get_item(ast, 2), "unary_expr_scalar");
        return nullptr;
    }

    bool MLIRGenImpl::is_unary_expr_scalar(const boost::property_tree::ptree& ast){
        std::cout << "is_unary_expr_scalar" << std::endl;
        return ast.count("unary_expr_scalar");
    }

    bool MLIRGenImpl::is_binary_expr_scalar(const boost::property_tree::ptree& ast){
        std::cout << "is_binary_expr_scalar" << std::endl;
        return ast.count("binary_expr_scalar");
    }

    mlir::Value MLIRGenImpl::parse_expr_scalar(const boost::property_tree::ptree& ast){
        std::cout <<    "parse_expr_scalar" << std::endl;
        auto expr_scalar = get_item(ast, 0);
        if(is_unary_expr_scalar(expr_scalar)){
            return parse_unary_expr_scalar( safe_get_child( expr_scalar, "unary_expr_scalar"));
        }else if(is_binary_expr_scalar(expr_scalar)){
            return parse_binary_expr_scalar( safe_get_child( expr_scalar, "binary_expr_scalar"));
        }
    }

    std::vector<mlir::Value> MLIRGenImpl::parse_array_1d(const boost::property_tree::ptree& ast){
        std::cout << "parse_array_1d" << std::endl;
        std::vector<mlir::Value> values;
        for (const auto& pair : ast) {
            if(pair.second.count("text") && (
                safe_get_str(pair.second, "text")=="[" or 
                safe_get_str(pair.second, "text")=="]" or
                safe_get_str(pair.second, "text")=="," )){
                continue;
            }
            values.push_back(parse_expr_scalar( safe_get_child( pair.second, "expr_scalar")));
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