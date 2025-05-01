#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include "cim/Dialect.h"
#include "cim/Parser.h"
#include "common/macros.h"
#include "mlir/IR/Operation.h"

static const boost::property_tree::ptree &
get_item(const boost::property_tree::ptree &ast, int index) {
  auto it = ast.begin();
  std::advance(it, index);
  return it->second;
}

std::string MLIRGenImpl::safe_get_str(const boost::property_tree::ptree &ast,
                                      const std::string &key) {
  if (ast.count(key)) {
    return ast.get<std::string>(key);
  } else {
    // tell user
    mlir::emitError(mlir::UnknownLoc::get(builder.getContext()),
                    "[safe_get_str] Key error: " + key);
    std::exit(1);
    return "";
  }
}

const boost::property_tree::ptree &
MLIRGenImpl::safe_get_child(const boost::property_tree::ptree &ast,
                            const std::string &key) {
  if (ast.count(key)) {
    return ast.get_child(key);
  } else {
    // tell user
    mlir::emitError(mlir::UnknownLoc::get(builder.getContext()),
                    "[safe_get_child] Key error: " + key);
    std::exit(1);
    return ast;
  }
}

MLIRGenImpl::MLIRGenImpl(mlir::MLIRContext &context)
    : builder(&context), loc(builder.getUnknownLoc()) {

  // GLOBAL_MEMORY = builder.getStringAttr("global");
  // LOCAL_MEMORY = builder.getStringAttr("local");
  // MACRO_MEMORY = builder.getStringAttr("macro");
  // loc = builder.getUnknownLoc();
}

mlir::ModuleOp MLIRGenImpl::parseJson(std::string json_path) {
  // Parse the module
  boost::property_tree::ptree ast;
  boost::property_tree::read_json(json_path, ast);
  return parseModule(ast);
}

mlir::ModuleOp
MLIRGenImpl::parseModule(const boost::property_tree::ptree &ast) {
  // Parse the module
  mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());

  auto program = safe_get_child(ast, "program");
  auto define_func_list = get_item(program, 0);

  mlir::Block *module_body = module.getBody();
  block_stack.push(module_body);
  builder.setInsertionPointToEnd(module_body);

  for (const auto &pair : program) {
    auto ast_define_func = safe_get_child(pair.second, "define_function");
    parse_func(ast_define_func);
  }

  block_stack.pop();
  return module;
}

void MLIRGenImpl::init_func_in_sign_table(const std::string &func_name) {
  current_func_name = func_name;
  signature_table[func_name] = std::unordered_map<std::string, mlir::Value>();
}

void MLIRGenImpl::add_to_sign_table(const std::string &arg_name,
                                    mlir::Value arg) {
  signature_table[current_func_name][arg_name] = arg;
}

mlir::Value MLIRGenImpl::get_from_sign_table(const std::string &arg_name) {
  if (signature_table.count(current_func_name)) {
    if (signature_table[current_func_name].count(arg_name)) {
      return signature_table[current_func_name][arg_name];
    } else {
      // raise: not support yet
      mlir::emitError(mlir::UnknownLoc::get(builder.getContext()),
                      "Variable not declare: " + arg_name);
      std::exit(1);
      return nullptr;
    }
  } else {
    // raise: not support yet
    mlir::emitError(mlir::UnknownLoc::get(builder.getContext()),
                    "Function not declare: " + current_func_name);
    std::exit(1);
    return nullptr;
  }
}

void MLIRGenImpl::add_func_to_sign_table(const std::string &func_name,
                                         mlir::func::FuncOp func) {
  signature_table_func[func_name] = func;
}

mlir::func::FuncOp
MLIRGenImpl::get_func_from_sign_table(const std::string &func_name) {
  if (signature_table_func.count(func_name)) {
    return signature_table_func[func_name];
  } else {
    // fail
    // raise: not support yet
    mlir::emitError(mlir::UnknownLoc::get(builder.getContext()),
                    "Function not declare: " + func_name);
    std::exit(1);
    return nullptr;
  }
}

void MLIRGenImpl::parse_func(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_func";

  const std::string func_name = safe_get_str(get_item(ast, 1), "text");
  init_func_in_sign_table(func_name);

  // Parse function args(each is mlir::Type)
  auto args =
      parse_func_args(safe_get_child(get_item(ast, 3), "func_param_list"));
  std::vector<mlir::Type> args_types = args.first;
  std::vector<std::string> args_names = args.second;

  // Parse function return type
  // return null for now.
  auto ret_type = builder.getIndexType();

  // Make function node
  auto func_type = builder.getFunctionType(args_types, {ret_type});
  auto func = builder.create<mlir::func::FuncOp>(loc, func_name, func_type);
  if (func_name != "main") {
    func.setPrivate();
  }
  mlir::Block *func_body = func.addEntryBlock();

  // Signature table
  for (int i = 0; i < args_names.size(); i++) {
    std::string name = args_names[i];
    mlir::Value arg = func_body->getArgument(i);
    add_to_sign_table(name, arg);
  }

  // Parse function body
  // std::advance(it, 3);
  // auto current_position = builder.getInsertionPoint();
  block_stack.push(func_body);
  builder.setInsertionPointToStart(func_body);

  parse_func_body(safe_get_child(get_item(ast, 6), "func_body"));

  block_stack.pop();
  builder.setInsertionPointToEnd(block_stack.top());

  // mlir::Value func_arg0 = func_body->getArgument(0);
  // mlir::Value func_arg1 = func_body->getArgument(1);
  // llvm::ArrayRef<int64_t> shape = {3,3,1,1};
  // mlir::Value a = builder.create<mlir::tensor::EmptyOp>(loc, shape,
  // builder.getI32Type()); mlir::Value b =
  // builder.create<mlir::cim::VVAddOp>(loc, func_arg0, func_arg1); mlir::Value
  // c = builder.create<mlir::cim::VVAddOp>(loc, a, func_arg1);

  add_func_to_sign_table(func_name, func);

  LOG_DEBUG << "parse_func finish.";
}

void MLIRGenImpl::parse_func_body(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_func_body";

  parse_stmt_list(safe_get_child(get_item(ast, 0), "stmt_list"));

  // Check if the block already has a terminator
  if (!builder.getInsertionBlock()->mightHaveTerminator()) {
    mlir::Value zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    builder.create<mlir::func::ReturnOp>(loc, zero);
  }

  LOG_DEBUG << "parse_func_body finish.";
}

void MLIRGenImpl::parse_stmt_list(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_stmt_list";

  for (const auto &pair : ast) {
    parse_stmt(safe_get_child(pair.second, "stmt"));
  }
}

void MLIRGenImpl::parse_stmt(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_stmt";
  auto ast_stmt = get_item(ast, 0);
  if (is_assign_stmt(ast_stmt)) {
    parse_assign_stmt(safe_get_child(ast_stmt, "stmt_assign"));
  } else if (is_return_stmt(ast_stmt)) {
    parse_return_stmt(safe_get_child(ast_stmt, "stmt_return"));
  } else if (is_call_stmt(ast_stmt)) {
    parse_call_stmt(safe_get_child(ast_stmt, "stmt_call"));
  } else if (is_for_stmt(ast_stmt)) {
    parse_for_stmt(safe_get_child(ast_stmt, "stmt_for"));
  } else if (is_if_else_stmt(ast_stmt)) {
    parse_if_else_stmt(safe_get_child(ast_stmt, "stmt_if_else"));
  } else {
    // raise: not support yet
    mlir::emitError(mlir::UnknownLoc::get(builder.getContext()),
                    "Not support stmt: " + ast.begin()->first);
    std::exit(1);
  }
  LOG_DEBUG << "parse_stmt finish";
}

bool MLIRGenImpl::is_assign_stmt(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "is_assign_stmt";

  return ast.begin()->first == "stmt_assign";
}
bool MLIRGenImpl::is_return_stmt(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "is_return_stmt";

  return ast.begin()->first == "stmt_return";
}
bool MLIRGenImpl::is_call_stmt(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "is_call_stmt";

  return ast.begin()->first == "stmt_call";
}
bool MLIRGenImpl::is_for_stmt(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "is_for_stmt";

  return ast.count("stmt_for");
}
bool MLIRGenImpl::is_if_else_stmt(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "is_if_else_stmt";

  return ast.count("stmt_if_else");
}
/*
    Stmt :
        stmt_assign,
        stmt_call,
        stmt_for,
        stmt_return

*/

void MLIRGenImpl::parse_call_stmt(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_call_stmt";
  parse_call(safe_get_child(get_item(ast, 0), "call"));
}

void MLIRGenImpl::parse_assign_stmt(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_assign_stmt";
  // LHS
  std::string var_name = safe_get_str(get_item(ast, 0), "text");

  // RHS
  mlir::Value expr = parse_expr(safe_get_child(get_item(ast, 2), "expr"));

  // Add to sign table
  add_to_sign_table(var_name, expr);
}

void MLIRGenImpl::parse_return_stmt(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_return_stmt";
  auto ast_expr = safe_get_child(get_item(ast, 1), "expr");
  mlir::Value expr = parse_expr(ast_expr);
  builder.create<mlir::func::ReturnOp>(loc, expr);
}

void MLIRGenImpl::parse_for_stmt(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_for_stmt";

  int tag = 0;
  // check unroll tag
  if (get_item(ast, 0).count("unroll")) {
    // unroll
    LOG_DEBUG << "unroll";
    tag = 1;
  }

  std::string iter_var_name = safe_get_str(get_item(ast, tag + 1), "text");
  std::vector<mlir::Value> range =
      parse_for_range(safe_get_child(get_item(ast, tag + 3), "for_range"));
  mlir::Value range_begin = range[0];
  mlir::Value range_end = range[1];
  mlir::Value range_step = range[2];

  // loop-carried variables
  auto loop_carried_names_and_variables =
      parse_carry(safe_get_child(get_item(ast, tag + 4), "carry"));
  auto loop_carried_names = loop_carried_names_and_variables.first;
  auto loop_carried_variables = loop_carried_names_and_variables.second;
  mlir::scf::ForOp for_op = builder.create<mlir::scf::ForOp>(
      loc, range_begin, range_end, range_step, loop_carried_variables);

  // mark for_op tag
  if (tag == 1) {
    // unrollForOps.push_back(for_op);
    for_op->setAttr("unroll", builder.getUnitAttr());
  }

  // Add to sign table
  llvm::SmallVector<mlir::Value> for_args;
  for (const auto &barg : llvm::enumerate(for_op.getBody(0)->getArguments())) {
    for_args.push_back(barg.value());
  }
  add_to_sign_table(iter_var_name, for_args[0]);
  for (int i = 1; i < for_args.size(); i++) { // for_args[0] is iter var
    add_to_sign_table(loop_carried_names[i - 1], for_args[i]);
  }

  // Loop Body
  mlir::Block *for_body = for_op.getBody();
  block_stack.push(for_body);
  builder.setInsertionPointToStart(for_op.getBody());

  parse_stmt_list(safe_get_child(get_item(ast, tag + 6), "stmt_list"));
  // yield
  auto yield_variables =
      parse_carry(safe_get_child(get_item(ast, tag + 4), "carry")).second;
  if (yield_variables.size() > 0) {
    builder.create<mlir::scf::YieldOp>(loc, yield_variables);
  }

  block_stack.pop();
  if (block_stack.top()->mightHaveTerminator()){
    if (auto yieldOp = mlir::dyn_cast_or_null<mlir::scf::YieldOp>(block_stack.top()->getTerminator())){
      builder.setInsertionPoint(yieldOp);
    }else{
      builder.setInsertionPointToEnd(block_stack.top());
    }
  } else {
    builder.setInsertionPointToEnd(block_stack.top());
  }
  // builder.setInsertionPointToEnd(block_stack.top());

  // replace carry variables with new values
  mlir::ValueRange for_result = for_op.getResults();
  LOG_DEBUG << "for_result size: " << for_result.size();
  for (int i = 0; i < for_result.size(); i++) { // for_args[0] is iter var
    add_to_sign_table(loop_carried_names[i], for_result[i]);
  }


}

void MLIRGenImpl::parse_if_else_stmt(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_if_else_stmt";

  auto ast_expr = safe_get_child(get_item(ast, 2), "expr");
  mlir::Value cond = parse_expr(ast_expr);

  auto loop_carried_names_and_variables =
      parse_carry(safe_get_child(get_item(ast, 4), "carry"));
  auto loop_carried_names = loop_carried_names_and_variables.first;
  auto loop_carried_variables = loop_carried_names_and_variables.second;

  // Get result types from carried variables
  llvm::SmallVector<mlir::Type> resultTypes;
  for (auto var : loop_carried_variables) {
    resultTypes.push_back(var.getType());
  }

  // Create if operation with correct result types
  mlir::scf::IfOp ifOp = builder.create<mlir::scf::IfOp>(loc, resultTypes, cond,
                                             /*else=*/true);

  // build then body
  mlir::Block *then_body = &ifOp.getThenRegion().front();
  block_stack.push(then_body);
  builder.setInsertionPointToStart(then_body);

  parse_stmt_list(safe_get_child(get_item(ast, 6), "stmt_list"));
  auto yield_variables =
      parse_carry(safe_get_child(get_item(ast, 4), "carry")).second;
  builder.create<mlir::scf::YieldOp>(loc, yield_variables);

  block_stack.pop();
  builder.setInsertionPointToEnd(block_stack.top());

  // build else body
  mlir::Block *else_body = &ifOp.getElseRegion().front();
  block_stack.push(else_body);
  builder.setInsertionPointToStart(else_body);

  parse_stmt_list(safe_get_child(get_item(ast, 10), "stmt_list"));
  auto yield_variables2 =
      parse_carry(safe_get_child(get_item(ast, 4), "carry")).second;
  builder.create<mlir::scf::YieldOp>(loc, yield_variables2);

  block_stack.pop();
  builder.setInsertionPointToEnd(block_stack.top());

  // replace carry variables with new values
  mlir::ValueRange if_result = ifOp.getResults();
  for (int i = 0; i < if_result.size(); i++) { // for_args[0] is iter var
    add_to_sign_table(loop_carried_names[i], if_result[i]);
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

std::pair<llvm::SmallVector<std::string>, llvm::SmallVector<mlir::Value>>
MLIRGenImpl::parse_carry(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_carry";
  auto ast_carry_list = safe_get_child(get_item(ast, 2), "carry_list");
  std::pair<llvm::SmallVector<std::string>, llvm::SmallVector<mlir::Value>>
      carry_list = parse_carry_list(ast_carry_list);
  LOG_DEBUG << "parse_carry finish";
  return carry_list;
}
std::pair<llvm::SmallVector<std::string>, llvm::SmallVector<mlir::Value>>
MLIRGenImpl::parse_carry_list(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_carry_list";

  llvm::SmallVector<std::string> var_name_list;
  llvm::SmallVector<mlir::Value> vec_carry_list;
  for (const auto &pair : ast) {
    if (pair.second.count("var")) {
      auto ast_var = safe_get_child(pair.second, "var");
      auto var_and_name = parse_var_and_name(ast_var);
      var_name_list.push_back(var_and_name.first);
      vec_carry_list.push_back(var_and_name.second);
    }
  }
  // mlir::ValueRange carry_list(vec_carry_list);
  LOG_DEBUG << "parse_carry_list finish " << var_name_list.size() << ", " << vec_carry_list.size();
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

std::vector<mlir::Value>
MLIRGenImpl::parse_for_range(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_for_range";

  std::vector<mlir::Value> range_values;
  auto ast_for = get_item(ast, 0);
  if (ast_for.count("for_range_1")) {
    range_values = parse_for_range_1(safe_get_child(ast_for, "for_range_1"));
  } else if (ast_for.count("for_range_2")) {
    range_values = parse_for_range_2(safe_get_child(ast_for, "for_range_2"));
  } else if (ast_for.count("for_range_3")) {
    range_values = parse_for_range_3(safe_get_child(ast_for, "for_range_3"));
  } else {
    mlir::emitError(mlir::UnknownLoc::get(builder.getContext()),
                    "Not support range");
    std::exit(1);
  }
  LOG_DEBUG << "parse_range finish";
  return range_values;
}

std::vector<mlir::Value>
MLIRGenImpl::parse_for_range_1(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_for_range_1";

  mlir::Value begin = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  mlir::Value end =
      parse_const_or_var(safe_get_child(get_item(ast, 1), "const_or_var"));
  mlir::Value stride = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);

  LOG_DEBUG << "parse_for_range_1 finish";
  return {begin, end, stride};
}

std::vector<mlir::Value>
MLIRGenImpl::parse_for_range_2(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_for_range_2";

  mlir::Value begin =
      parse_const_or_var(safe_get_child(get_item(ast, 1), "const_or_var"));
  mlir::Value end =
      parse_const_or_var(safe_get_child(get_item(ast, 3), "const_or_var"));
  mlir::Value stride = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);

  LOG_DEBUG << "parse_for_range_2 finish";
  return {begin, end, stride};
}

std::vector<mlir::Value>
MLIRGenImpl::parse_for_range_3(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_for_range_3";

  mlir::Value begin =
      parse_const_or_var(safe_get_child(get_item(ast, 1), "const_or_var"));
  mlir::Value end =
      parse_const_or_var(safe_get_child(get_item(ast, 3), "const_or_var"));
  mlir::Value stride =
      parse_const_or_var(safe_get_child(get_item(ast, 5), "const_or_var"));

  LOG_DEBUG << "parse_for_range_3 finish";
  return {begin, end, stride};
}

/*
 Range end
*/

mlir::Value
MLIRGenImpl::parse_call_return_value(const boost::property_tree::ptree &ast) {
  // call a function, and get return value

  LOG_DEBUG << "parse_call_return_value";
  std::string call_func_name = safe_get_str(get_item(ast, 0), "text");

  if (call_func_name == "Shape") {
    return parse_builtin_shape(ast);
  } else if (call_func_name == "Slice") {
    return parse_builtin_slice(ast);
  } else if (call_func_name == "Buffer") {
    return parse_builtin_buffer(ast);
  } else if (call_func_name == "Load") {
    return parse_builtin_load(ast);
  } else if (call_func_name == "Min") {
    return parse_builtin_min(ast);
  } else if (call_func_name == "Max") {
    return parse_builtin_max(ast);
  } else if (call_func_name == "Addr") {
    return parse_builtin_addr(ast);
  } else if (call_func_name == "Select") {
    return parse_builtin_select(ast);
  }

  // check sign table
  mlir::ValueRange param_list = parse_call_param_list(
      safe_get_child(get_item(ast, 2), "call_param_list"));
  mlir::func::FuncOp func = get_func_from_sign_table(call_func_name);
  mlir::func::CallOp call =
      builder.create<mlir::func::CallOp>(loc, func, param_list);

  return call.getResult(0);
}

void MLIRGenImpl::parse_call(const boost::property_tree::ptree &ast) {
  // call a function, and get return value

  LOG_DEBUG << "parse_call";
  std::string call_func_name = safe_get_str(get_item(ast, 0), "text");

  if (call_func_name == "Trans") {
    parse_builtin_trans(ast);
    return;
  } else if (call_func_name == "SIMD") {
    parse_builtin_simd(ast);
    return;
  }  else if (call_func_name == "Print") {
    parse_builtin_print(ast);
    return;
  } else if (call_func_name == "Debug") {
    parse_builtin_debug(ast);
    return;
  } else if (call_func_name == "Free") {
    parse_builtin_free(ast);
    return;
  } else if (call_func_name == "CIMComputeDense") {
    parse_builtin_cimcompute_dense(ast);
    return;
  } else if (call_func_name == "CIMComputeValueSparse") {
    parse_builtin_cimcompute_value_sparse(ast);
    return;
  } else if (call_func_name == "CIMComputeBitSparse") {
    parse_builtin_cimcompute_bit_sparse(ast);
    return;
  } else if (call_func_name == "CIMComputeValueBitSparse") {
    parse_builtin_cimcompute_value_bit_sparse(ast);
    return;
  } else if (call_func_name == "CIMOutput") {
    parse_builtin_cimoutput(ast);
    return;
  } else if (call_func_name == "CIMOutputSum") {
    parse_builtin_cimoutput_sum(ast);
    return;
  } else if (call_func_name == "CIMTransfer") {
    parse_builtin_cimtransfer(ast);
    return;
  } else if (call_func_name == "CIMSet") {
    parse_builtin_cimset(ast);
    return;
  } else if (call_func_name == "Save") {
    parse_builtin_save(ast);
    return;
  } else if (call_func_name == "SpecialRegSet") {
    parse_builtin_special_reg_set(ast);
    return;
  } else if (call_func_name == "Send") {
    parse_builtin_send(ast);
    return;
  } else if (call_func_name == "Recv") {
    parse_builtin_recv(ast);
    return;
  }

  // check sign table
  mlir::func::FuncOp func = get_func_from_sign_table(call_func_name);
  llvm::SmallVector<mlir::Value> param_list = parse_call_param_list(
      safe_get_child(get_item(ast, 2), "call_param_list"));

  auto fnType = func.getFunctionType();
  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i) {
    auto paramType = fnType.getInput(i);
    auto callType = param_list[i].getType();
    if (paramType != callType) {
      // cast param_list[i] to paramType
      // check if paramType is a memref type
      if (paramType.isa<mlir::MemRefType>() 
      || paramType.isa<mlir::UnrankedMemRefType>()) {
        param_list[i] = builder.create<mlir::memref::CastOp>(loc, paramType, param_list[i]);
      } else {
        // raise: not support yet
        std::string message;
        llvm::raw_string_ostream os(message);
        os << "param type mismatch: " << paramType << " vs " << callType << "\n";
        std::cerr << os.str();
        std::exit(1);
      }
    }
  }
  
  auto call =
      builder.create<mlir::func::CallOp>(loc, func, param_list);
  LOG_DEBUG << "parse_call finish";
  return;
}

llvm::SmallVector<mlir::Value>
MLIRGenImpl::parse_call_param_list(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_call_param_list";
  llvm::SmallVector<mlir::Value> vec_param_list;
  for (const auto &pair : ast) {
    if (pair.second.count("call_param")) {
      auto ast_call_param = safe_get_child(pair.second, "call_param");
      vec_param_list.push_back(parse_call_param(ast_call_param));
    }
  }
  // mlir::ValueRange param_list(vec_param_list);
  LOG_DEBUG << "parse_call_param_list finish";
  return vec_param_list;
}

mlir::Value
MLIRGenImpl::parse_call_param(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_call_param";
  auto ast_expr = safe_get_child(get_item(ast, 0), "expr");
  return parse_expr(ast_expr);
}

/*
 * Builtin Functions Begin
 */

mlir::Value
MLIRGenImpl::parse_builtin_shape(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_builtin_shape";
  auto ast_param_list = safe_get_child(get_item(ast, 2), "call_param_list");

  auto ast_buffer = safe_get_child(get_item(ast_param_list, 0), "call_param");
  auto ast_index = safe_get_child(get_item(ast_param_list, 2), "call_param");

  mlir::Value buffer =
      parse_expr(safe_get_child(get_item(ast_buffer, 0), "expr"));
  mlir::Value index =
      parse_expr(safe_get_child(get_item(ast_index, 0), "expr"));
  return builder.create<mlir::cim::ShapeOp>(loc, buffer, index);
}

void MLIRGenImpl::parse_builtin_trans(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_builtin_trans";
  auto ast_param_list = safe_get_child(get_item(ast, 2), "call_param_list");

  auto ast_src = safe_get_child(get_item(ast_param_list, 0), "call_param");
  auto ast_dst = safe_get_child(get_item(ast_param_list, 2), "call_param");

  mlir::Value src = parse_expr(safe_get_child(get_item(ast_src, 0), "expr"));
  mlir::Value dst = parse_expr(safe_get_child(get_item(ast_dst, 0), "expr"));
  builder.create<mlir::cim::CopyOp>(loc, src, dst);
}

mlir::Value
MLIRGenImpl::parse_builtin_slice(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_builtin_slice";
  auto ast_param_list = safe_get_child(get_item(ast, 2), "call_param_list");

  auto ast_src = safe_get_child(get_item(ast_param_list, 0), "call_param");
  auto ast_offsets = safe_get_child(get_item(ast_param_list, 2), "call_param");
  auto ast_sizes = safe_get_child(get_item(ast_param_list, 4), "call_param");
  auto ast_strides = safe_get_child(get_item(ast_param_list, 6), "call_param");

  mlir::Value src = parse_expr(safe_get_child(get_item(ast_src, 0), "expr"));
  mlir::SmallVector<mlir::Value> offsets =
      parse_array_1d(safe_get_child(get_item(ast_offsets, 0), "array1d"));
  LOG_DEBUG << "parse_builtin_slice offsets finish";
  LOG_DEBUG << offsets.size();
  mlir::SmallVector<mlir::Value> sizes =
      parse_array_1d(safe_get_child(get_item(ast_sizes, 0), "array1d"));
  LOG_DEBUG << "parse_builtin_slice sizes finish";
  LOG_DEBUG << sizes.size();
  mlir::SmallVector<mlir::Value> strides =
      parse_array_1d(safe_get_child(get_item(ast_strides, 0), "array1d"));
  LOG_DEBUG << "parse_builtin_slice strides finish";
  LOG_DEBUG << strides.size();
  mlir::Value result = builder.create<mlir::memref::SubViewOp>(
      loc, src, cast_to_index_type(offsets), cast_to_index_type(sizes),
      cast_to_index_type(strides));
  LOG_DEBUG << "parse_builtin_slice finish";
  return result;
}


void MLIRGenImpl::parse_builtin_simd(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_builtin_simd";
  auto ast_param_list = safe_get_child(get_item(ast, 2), "call_param_list");
  
  
  int num_param = (ast_param_list.size() + 1) / 2;
  auto ast_op_id = safe_get_child(get_item(ast_param_list, 0), "call_param");
  mlir::Value op_id = parse_expr(safe_get_child(get_item(ast_op_id, 0), "expr"));

  mlir::SmallVector<mlir::Value> operands;
  for (int i = 1; i < num_param - 1; i++) {
    auto ast_operand = safe_get_child(get_item(ast_param_list, 2 * i), "call_param");
    operands.push_back(parse_expr(safe_get_child(get_item(ast_operand, 0), "expr")));
  }
  auto ast_output = safe_get_child(get_item(ast_param_list, 2 * (num_param - 1)), "call_param");
  mlir::Value output = parse_expr(safe_get_child(get_item(ast_output, 0), "expr"));

  builder.create<mlir::cim::SIMDOp>(loc, op_id, operands, output);
}

mlir::Value
MLIRGenImpl::parse_builtin_buffer(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_builtin_buffer";
  auto ast_param_list = safe_get_child(get_item(ast, 2), "call_param_list");

  // Data type
  auto ast_dtype_call_param =
      safe_get_child(get_item(ast_param_list, 2), "call_param");
  auto ast_dtype =
      safe_get_child(get_item(ast_dtype_call_param, 0), "datatype");
  std::string str_dtype = safe_get_str(get_item(ast_dtype, 0), "text");
  mlir::Type dtype = parse_datatype(str_dtype);

  // Memory type
  auto ast_memory_call_param =
      safe_get_child(get_item(ast_param_list, 4), "call_param");
  auto ast_memory =
      safe_get_child(get_item(ast_memory_call_param, 0), "memory");
  std::string memory = safe_get_str(get_item(ast_memory, 0), "text");
  mlir::Attribute memory_attr = parse_device(memory);

  // Shape
  auto ast_shape = safe_get_child(get_item(ast_param_list, 0), "call_param");
  auto ast_shape_array1d =
      safe_get_child(get_item(ast_shape, 0), "const_array1d");
  std::vector<int64_t> shape = parse_const_array1d(ast_shape_array1d);

  mlir::MemRefType type =
      mlir::MemRefType::get(llvm::ArrayRef<int64_t>(shape), dtype,
                            mlir::MemRefLayoutAttrInterface(), memory_attr);
  mlir::memref::AllocOp alloc =
      builder.create<mlir::memref::AllocOp>(loc, type);
  buffer_type[alloc] = parse_memory_type(memory);
  return alloc.getResult();
}

void MLIRGenImpl::parse_builtin_print(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_builtin_print";
  auto ast_param_list = safe_get_child(get_item(ast, 2), "call_param_list");

  auto ast_value = safe_get_child(get_item(ast_param_list, 0), "call_param");
  mlir::Value value =
      parse_expr(safe_get_child(get_item(ast_value, 0), "expr"));

  // auto ast_comment = safe_get_child(get_item(ast_param_list,2),
  // "call_param"); std::string comment =
  // safe_get_str(safe_get_child(get_item(ast_comment, 0), "string"), "text");
  builder.create<mlir::cim::PrintOp>(loc, value);
}

void MLIRGenImpl::parse_builtin_debug(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_builtin_debug";
  builder.create<mlir::cim::DebugOp>(loc);
}

void MLIRGenImpl::parse_builtin_free(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_builtin_free";
  auto ast_param_list = safe_get_child(get_item(ast, 2), "call_param_list");

  auto ast_value = safe_get_child(get_item(ast_param_list, 0), "call_param");

  mlir::Value value =
      parse_expr(safe_get_child(get_item(ast_value, 0), "expr"));
  builder.create<mlir::memref::DeallocOp>(loc, value);
}

mlir::Value
MLIRGenImpl::parse_builtin_load(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_builtin_load";
  auto ast_param_list = safe_get_child(get_item(ast, 2), "call_param_list");

  auto ast_memref = safe_get_child(get_item(ast_param_list, 0), "call_param");
  auto ast_index = safe_get_child(get_item(ast_param_list, 2), "call_param");

  mlir::Value memref =
      parse_expr(safe_get_child(get_item(ast_memref, 0), "expr"));
  mlir::SmallVector<mlir::Value> _index =
      parse_array_1d(safe_get_child(get_item(ast_index, 0), "array1d"));
  mlir::SmallVector<mlir::Value> index = cast_to_index_type(_index);

  mlir::Value result = builder.create<mlir::memref::LoadOp>(loc, memref, index);
  return result;
}

mlir::Value
MLIRGenImpl::parse_builtin_min(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_builtin_min";
  auto ast_param_list = safe_get_child(get_item(ast, 2), "call_param_list");

  auto ast_lhs = safe_get_child(get_item(ast_param_list, 0), "call_param");
  auto ast_rhs = safe_get_child(get_item(ast_param_list, 2), "call_param");

  mlir::Value lhs = parse_expr(safe_get_child(get_item(ast_lhs, 0), "expr"));
  mlir::Value rhs = parse_expr(safe_get_child(get_item(ast_rhs, 0), "expr"));

  mlir::Value result = builder.create<mlir::arith::MinSIOp>(loc, lhs, rhs);
  return result;
}

mlir::Value
MLIRGenImpl::parse_builtin_max(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_builtin_max";
  auto ast_param_list = safe_get_child(get_item(ast, 2), "call_param_list");

  auto ast_lhs = safe_get_child(get_item(ast_param_list, 0), "call_param");
  auto ast_rhs = safe_get_child(get_item(ast_param_list, 2), "call_param");

  mlir::Value lhs = parse_expr(safe_get_child(get_item(ast_lhs, 0), "expr"));
  mlir::Value rhs = parse_expr(safe_get_child(get_item(ast_rhs, 0), "expr"));

  mlir::Value result = builder.create<mlir::arith::MaxSIOp>(loc, lhs, rhs);
  return result;
}

mlir::Value
MLIRGenImpl::parse_builtin_addr(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_builtin_addr";
  auto ast_param_list = safe_get_child(get_item(ast, 2), "call_param_list");

  auto ast_buf = safe_get_child(get_item(ast_param_list, 0), "call_param");

  mlir::Value buf = parse_expr(safe_get_child(get_item(ast_buf, 0), "expr"));

  mlir::Value result = builder.create<mlir::cim::AddrOp>(loc, buf);
  return result;
}

mlir::Value
MLIRGenImpl::parse_builtin_select(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_builtin_select";
  auto ast_param_list = safe_get_child(get_item(ast, 2), "call_param_list");

  auto ast_cond = safe_get_child(get_item(ast_param_list, 0), "call_param");
  auto ast_true = safe_get_child(get_item(ast_param_list, 2), "call_param");
  auto ast_false = safe_get_child(get_item(ast_param_list, 4), "call_param");

  mlir::Value cond = parse_expr(safe_get_child(get_item(ast_cond, 0), "expr"));
  mlir::Value true_ = parse_expr(safe_get_child(get_item(ast_true, 0), "expr"));
  mlir::Value false_ = parse_expr(safe_get_child(get_item(ast_false, 0), "expr"));

  mlir::Value result = builder.create<mlir::arith::SelectOp>(loc, cond, true_, false_);
  return result;
}

void MLIRGenImpl::parse_builtin_save(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_builtin_save";
  auto ast_param_list = safe_get_child(get_item(ast, 2), "call_param_list");

  auto ast_memref = safe_get_child(get_item(ast_param_list, 0), "call_param");
  auto ast_index = safe_get_child(get_item(ast_param_list, 2), "call_param");
  auto ast_value = safe_get_child(get_item(ast_param_list, 4), "call_param");

  mlir::Value memref =
      parse_expr(safe_get_child(get_item(ast_memref, 0), "expr"));
  mlir::Value value =
      parse_expr(safe_get_child(get_item(ast_value, 0), "expr"));
  mlir::SmallVector<mlir::Value> _index =
      parse_array_1d(safe_get_child(get_item(ast_index, 0), "array1d"));
  mlir::SmallVector<mlir::Value> index = cast_to_index_type(_index);

  builder.create<mlir::memref::StoreOp>(loc, value, memref, index);
  // return result;
}

void MLIRGenImpl::parse_builtin_cimcompute_dense(
    const boost::property_tree::ptree &ast) {
  parse_builtin_cimcompute(ast, false, false);
}

void MLIRGenImpl::parse_builtin_cimcompute_value_sparse(
    const boost::property_tree::ptree &ast) {
  parse_builtin_cimcompute(ast, true, false);
}

void MLIRGenImpl::parse_builtin_cimcompute_bit_sparse(
    const boost::property_tree::ptree &ast) {
  parse_builtin_cimcompute(ast, false, true);
}

void MLIRGenImpl::parse_builtin_cimcompute_value_bit_sparse(
    const boost::property_tree::ptree &ast) {
  parse_builtin_cimcompute(ast, true, true);
}

void MLIRGenImpl::parse_builtin_cimcompute(
    const boost::property_tree::ptree &ast, bool value_sparse,
    bool bit_sparse) {
  LOG_DEBUG << "parse_builtin_cimcompute";
  auto ast_param_list = safe_get_child(get_item(ast, 2), "call_param_list");

  auto ast_input = safe_get_child(get_item(ast_param_list, 0), "call_param");
  auto ast_macro = safe_get_child(get_item(ast_param_list, 2), "call_param");
  // auto ast_output = safe_get_child(get_item(ast_param_list,4), "call_param");

  mlir::Value input =
      parse_expr(safe_get_child(get_item(ast_input, 0), "expr"));
  mlir::Value macro =
      parse_expr(safe_get_child(get_item(ast_macro, 0), "expr"));
  // mlir::Value output = parse_expr(safe_get_child(get_item(ast_output, 0),
  // "expr"));

  mlir::IntegerAttr value_sparse_flag = mlir::IntegerAttr::get(
      builder.getIntegerType(1), llvm::APInt(1, value_sparse));
  mlir::IntegerAttr bit_sparse_flag = mlir::IntegerAttr::get(
      builder.getIntegerType(1), llvm::APInt(1, bit_sparse));
  builder.create<mlir::cim::CIMComputeOp>(loc, input, macro, value_sparse_flag,
                                          bit_sparse_flag);
}

void MLIRGenImpl::parse_builtin_cimoutput(
    const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_builtin_cimoutput";

  auto ast_param_list = safe_get_child(get_item(ast, 2), "call_param_list");

  auto ast_out_n = safe_get_child(get_item(ast_param_list, 0), "call_param");
  auto ast_out_mask = safe_get_child(get_item(ast_param_list, 2), "call_param");
  auto ast_output = safe_get_child(get_item(ast_param_list, 4), "call_param");

  mlir::Value out_n =
      parse_expr(safe_get_child(get_item(ast_out_n, 0), "expr"));
  mlir::Value out_mask =
      parse_expr(safe_get_child(get_item(ast_out_mask, 0), "expr"));
  mlir::Value output =
      parse_expr(safe_get_child(get_item(ast_output, 0), "expr"));

  builder.create<mlir::cim::CIMOutputOp>(loc, out_n, out_mask, output);
}

void MLIRGenImpl::parse_builtin_cimoutput_sum(
    const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_builtin_cimoutput_sum";

  auto ast_param_list = safe_get_child(get_item(ast, 2), "call_param_list");

  auto ast_out_n = safe_get_child(get_item(ast_param_list, 0), "call_param");
  auto ast_out_mask = safe_get_child(get_item(ast_param_list, 2), "call_param");
  auto ast_output_addr =
      safe_get_child(get_item(ast_param_list, 4), "call_param");

  mlir::Value out_n =
      parse_expr(safe_get_child(get_item(ast_out_n, 0), "expr"));
  mlir::Value out_mask =
      parse_expr(safe_get_child(get_item(ast_out_mask, 0), "expr"));
  mlir::Value output_addr =
      parse_expr(safe_get_child(get_item(ast_output_addr, 0), "expr"));

  builder.create<mlir::cim::CIMOutputSumOp>(loc, out_n, out_mask, output_addr);
}

void MLIRGenImpl::parse_builtin_cimtransfer(
    const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_builtin_cimtransfer";

  auto ast_param_list = safe_get_child(get_item(ast, 2), "call_param_list");

  auto ast_output = safe_get_child(get_item(ast_param_list, 0), "call_param");
  auto ast_output_num =
      safe_get_child(get_item(ast_param_list, 2), "call_param");
  auto ast_output_mask =
      safe_get_child(get_item(ast_param_list, 4), "call_param");
  auto ast_buffer = safe_get_child(get_item(ast_param_list, 6), "call_param");
  auto ast_dst = safe_get_child(get_item(ast_param_list, 8), "call_param");

  mlir::Value src = parse_expr(safe_get_child(get_item(ast_output, 0), "expr"));
  mlir::Value output_num =
      parse_expr(safe_get_child(get_item(ast_output_num, 0), "expr"));
  mlir::Value output_mask =
      parse_expr(safe_get_child(get_item(ast_output_mask, 0), "expr"));
  mlir::Value buffer =
      parse_expr(safe_get_child(get_item(ast_buffer, 0), "expr"));
  mlir::Value dst = parse_expr(safe_get_child(get_item(ast_dst, 0), "expr"));

  builder.create<mlir::cim::CIMTransferOp>(loc, src, output_num, output_mask,
                                           buffer, dst);
}

void MLIRGenImpl::parse_builtin_cimset(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_builtin_cimset";

  auto ast_param_list = safe_get_child(get_item(ast, 2), "call_param_list");

  auto ast_mask = safe_get_child(get_item(ast_param_list, 0), "call_param");

  mlir::Value mask = parse_expr(safe_get_child(get_item(ast_mask, 0), "expr"));

  builder.create<mlir::cim::CIMSetOp>(loc, mask);
}

void MLIRGenImpl::parse_builtin_special_reg_set(
    const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_builtin_special_reg_set";
  auto ast_param_list = safe_get_child(get_item(ast, 2), "call_param_list");

  auto ast_special_reg =
      safe_get_child(get_item(ast_param_list, 0), "call_param");
  auto ast_set_value =
      safe_get_child(get_item(ast_param_list, 2), "call_param");

  mlir::Value special_reg =
      parse_expr(safe_get_child(get_item(ast_special_reg, 0), "expr"));
  mlir::Value set_value =
      parse_expr(safe_get_child(get_item(ast_set_value, 0), "expr"));
  builder.create<mlir::cim::SpecialRegSetOp>(loc, special_reg, set_value);
}

void MLIRGenImpl::parse_builtin_send(
    const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_builtin_special_send";
  auto ast_param_list = safe_get_child(get_item(ast, 2), "call_param_list");

  auto ast_src_buffer =
      safe_get_child(get_item(ast_param_list, 0), "call_param");
  auto ast_dst_buffer =
      safe_get_child(get_item(ast_param_list, 2), "call_param");
  auto ast_core_id =
      safe_get_child(get_item(ast_param_list, 4), "call_param");
  auto ast_transfer_id =
    safe_get_child(get_item(ast_param_list, 6), "call_param");

  mlir::Value src_buffer = parse_expr(safe_get_child(get_item(ast_src_buffer, 0), "expr"));
  mlir::Value dst_buffer = parse_expr(safe_get_child(get_item(ast_dst_buffer, 0), "expr"));
  mlir::Value core_id = parse_expr(safe_get_child(get_item(ast_core_id, 0), "expr"));
  mlir::Value transfer_id = parse_expr(safe_get_child(get_item(ast_transfer_id, 0), "expr"));
  builder.create<mlir::cim::SendOp>(loc, src_buffer, dst_buffer, core_id, transfer_id);
}


void MLIRGenImpl::parse_builtin_recv(
    const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_builtin_special_recv";
  auto ast_param_list = safe_get_child(get_item(ast, 2), "call_param_list");

  auto ast_src_buffer =
      safe_get_child(get_item(ast_param_list, 0), "call_param");
  auto ast_dst_buffer =
      safe_get_child(get_item(ast_param_list, 2), "call_param");
  auto ast_core_id =
      safe_get_child(get_item(ast_param_list, 4), "call_param");
  auto ast_transfer_id =
    safe_get_child(get_item(ast_param_list, 6), "call_param");

  mlir::Value src_buffer = parse_expr(safe_get_child(get_item(ast_src_buffer, 0), "expr"));
  mlir::Value dst_buffer = parse_expr(safe_get_child(get_item(ast_dst_buffer, 0), "expr"));
  mlir::Value core_id = parse_expr(safe_get_child(get_item(ast_core_id, 0), "expr"));
  mlir::Value transfer_id = parse_expr(safe_get_child(get_item(ast_transfer_id, 0), "expr"));
  builder.create<mlir::cim::RecvOp>(loc, src_buffer, dst_buffer, core_id, transfer_id);
}

/*
 * Built-in Functions End
 */

// mlir::Value parse_var(const boost::property_tree::ptree& ast){
//     auto constant = get_item(ast, 0);
//     int value = std::stoi(constant.get<std::string>("text"));
// }

mlir::Value MLIRGenImpl::parse_var(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_var";
  std::string var_name = safe_get_str(get_item(ast, 0), "text");
  mlir::Value var = get_from_sign_table(var_name);
  return var;
}

std::pair<std::string, mlir::Value>
MLIRGenImpl::parse_var_and_name(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_var";
  std::string var_name = safe_get_str(get_item(ast, 0), "text");
  mlir::Value var = get_from_sign_table(var_name);
  return std::make_pair(var_name, var);
}

int64_t MLIRGenImpl::parse_const_int(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_const_int";

  auto const_node = get_item(ast, 0);
  int value = std::stoi(safe_get_str(const_node, "text"));
  return static_cast<int64_t>(value);
}

mlir::Value MLIRGenImpl::parse_const(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_const";

  auto const_node = get_item(ast, 0);
  int value = std::stoi(safe_get_str(const_node, "text"));
  mlir::Value const_value =
      builder.create<mlir::arith::ConstantIndexOp>(loc, value);
  LOG_DEBUG << "parse_const finish";
  return const_value;
}

bool MLIRGenImpl::is_const(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "is_const";
  return ast.count("constant");
}

bool MLIRGenImpl::is_var(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "is_var";
  return ast.count("var");
}

mlir::Value
MLIRGenImpl::parse_const_or_var(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_const_or_var";

  auto const_or_var = get_item(ast, 0);
  if (is_const(const_or_var)) {
    return parse_const(safe_get_child(const_or_var, "constant"));
  } else if (is_var(const_or_var)) {
    return parse_var(safe_get_child(const_or_var, "var"));
  }
}

bool MLIRGenImpl::is_const_or_var(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "is_const_or_var";
  return ast.count("const_or_var");
}

bool MLIRGenImpl::is_call(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "is_call";
  return ast.count("call");
}


mlir::Value MLIRGenImpl::parse_buffer_slice(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_buffer_slice";
  auto ast_var = safe_get_child(get_item(ast, 0), "var");
  auto var = parse_var(ast_var);

  auto ast_slice_list = safe_get_child(get_item(ast, 2), "slice_list");
  auto offsets_sizes_strides = parse_slice_list(var, ast_slice_list);

  mlir::SmallVector<mlir::Value> offsets = std::get<0>(offsets_sizes_strides);
  mlir::SmallVector<mlir::Value> sizes = std::get<1>(offsets_sizes_strides);
  mlir::SmallVector<mlir::Value> strides = std::get<2>(offsets_sizes_strides);

  mlir::Value buffer = builder.create<mlir::memref::SubViewOp>(loc, var, offsets, sizes, strides);
  return buffer;
}

std::tuple<mlir::SmallVector<mlir::Value>, mlir::SmallVector<mlir::Value>, mlir::SmallVector<mlir::Value>>
MLIRGenImpl::parse_slice_list(mlir::Value var, const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_slice_list";
  mlir::SmallVector<mlir::Value> offsets;
  mlir::SmallVector<mlir::Value> sizes;
  mlir::SmallVector<mlir::Value> strides;
  // iter over ast
  int slice_id = 0;
  for (const auto &pair : ast) {
    if (is_slice(pair.second)) {
      auto ast_slice = safe_get_child(pair.second, "slice");
      mlir::Value slice_offset_value;
      mlir::Value slice_len_value;
      if (get_item(ast_slice, 0).count("slice_range")) {
        auto ast_slice_range = safe_get_child(get_item(ast_slice, 0), "slice_range");
        auto ast_slice_offset = safe_get_child(get_item(ast_slice_range, 0), "slice_offset");
        auto ast_slice_end = safe_get_child(get_item(ast_slice_range, 2), "slice_end");
        
        if (ast_slice_offset.empty()) {
          slice_offset_value = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
        } else {
          slice_offset_value = parse_expr(safe_get_child(get_item(ast_slice_offset, 0), "expr"));
        }
        mlir::Value slice_end_value;
        if (ast_slice_end.empty()) {
          mlir::Value slice_index = builder.create<mlir::arith::ConstantIndexOp>(loc, slice_id);
          slice_end_value = builder.create<mlir::cim::ShapeOp>(loc, var, slice_index);
        } else {
          slice_end_value = parse_expr(safe_get_child(get_item(ast_slice_end, 0), "expr"));
        }
        slice_len_value = builder.create<mlir::arith::SubIOp>(loc, slice_end_value, slice_offset_value);
      } else if (get_item(ast_slice, 0).count("slice_scalar")) {
        auto ast_slice_scalar = safe_get_child(get_item(ast_slice, 0), "slice_scalar");
        slice_offset_value = parse_expr(safe_get_child(get_item(ast_slice_scalar, 0), "expr"));
        slice_len_value = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
      } else {
        // raise: not support yet
        mlir::emitError(mlir::UnknownLoc::get(builder.getContext()),
                        "Not support slice: " + ast.begin()->first);
        std::exit(1);
      }

      offsets.push_back(slice_offset_value);
      sizes.push_back(slice_len_value);
      strides.push_back(builder.create<mlir::arith::ConstantIndexOp>(loc, 1));

      slice_id += 1;

    }
  }
  return std::make_tuple(offsets, sizes, strides);
}

bool MLIRGenImpl::is_buffer_slice(const boost::property_tree::ptree &ast) {
  return ast.count("buffer_slice");
}

bool MLIRGenImpl::is_slice(const boost::property_tree::ptree &ast) {
  return ast.count("slice");
}

mlir::Value
MLIRGenImpl::parse_unary_expr(const boost::property_tree::ptree &ast) {
  /*
      unary_expr: call | const_or_var;
  */
  LOG_DEBUG << "parse_unary_expr";
  auto unary_expr = get_item(ast, 0);
  if (is_const_or_var(unary_expr)) {
    return parse_const_or_var(safe_get_child(unary_expr, "const_or_var"));
  } else if (is_call(unary_expr)) {
    return parse_call_return_value(safe_get_child(unary_expr, "call"));
  } else if (is_buffer_slice(unary_expr)) {
    return parse_buffer_slice(safe_get_child(unary_expr, "buffer_slice"));
  } else {
    // raise: not support yet
    mlir::emitError(mlir::UnknownLoc::get(builder.getContext()),
                    "Not support unary_expr: " + ast.begin()->first);
    std::exit(1);
    return nullptr;
  }
}

mlir::Value
MLIRGenImpl::parse_binary_expr(const boost::property_tree::ptree &ast) {
  /*
      binary_expr: unary_expr BINARY_OP unary_expr;
  */
  LOG_DEBUG << "parse_binary_expr";
  auto ast_lhs = safe_get_child(get_item(ast, 0), "unary_expr");
  mlir::Value lhs = parse_unary_expr(ast_lhs);

  auto binary_op = safe_get_str(get_item(ast, 1), "text");

  auto ast_rhs = safe_get_child(get_item(ast, 2), "unary_expr");
  mlir::Value rhs = parse_unary_expr(ast_rhs);

  if (binary_op == "+") {
    return builder.create<mlir::arith::AddIOp>(loc, lhs, rhs);
  } else if (binary_op == "-") {
    return builder.create<mlir::arith::SubIOp>(loc, lhs, rhs);
  } else if (binary_op == "*") {
    return builder.create<mlir::arith::MulIOp>(loc, lhs, rhs);
  } else if (binary_op == "/") {
    return builder.create<mlir::arith::DivSIOp>(loc, lhs, rhs);
  } else if (binary_op == "%") {
    return builder.create<mlir::arith::RemSIOp>(loc, lhs, rhs);
  } else if (binary_op == "==") {
    return builder.create<mlir::arith::CmpIOp>(loc,mlir::arith::CmpIPredicate::eq, lhs, rhs);
  } else if (binary_op == "!=") {
    return builder.create<mlir::arith::CmpIOp>(loc,mlir::arith::CmpIPredicate::ne, lhs, rhs);
  }else if (binary_op == "<=") {
    return builder.create<mlir::arith::CmpIOp>(loc,mlir::arith::CmpIPredicate::sle, lhs, rhs);
  }else if (binary_op == "<<") {
    return builder.create<mlir::arith::CmpIOp>(loc,mlir::arith::CmpIPredicate::slt, lhs, rhs);
  }else if (binary_op == ">=") {
    return builder.create<mlir::arith::CmpIOp>(loc,mlir::arith::CmpIPredicate::sge, lhs, rhs);
  }else if (binary_op == ">>") {
    return builder.create<mlir::arith::CmpIOp>(loc,mlir::arith::CmpIPredicate::sgt, lhs, rhs);
  }else if (binary_op == "&&") {
    return builder.create<mlir::arith::AndIOp>(loc, lhs, rhs);
  }else {
    // raise: not support yet
    mlir::emitError(mlir::UnknownLoc::get(builder.getContext()),
                    "Not support binary_op: " + binary_op);
    std::exit(1);
    return nullptr;
  }
}

bool MLIRGenImpl::is_unary_expr(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "is_unary_expr";
  return ast.count("unary_expr");
}

bool MLIRGenImpl::is_binary_expr(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "is_binary_expr";
  return ast.count("binary_expr");
}

mlir::Value MLIRGenImpl::parse_expr(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_expr";
  auto expr = get_item(ast, 0);
  if (is_condition_expr(expr)) {
    return parse_condition_expr(safe_get_child(expr, "condition_expr"));
  } else {
    // raise: not support yet
    mlir::emitError(mlir::UnknownLoc::get(builder.getContext()),
                    "Not support expr: " + ast.begin()->first);
    std::exit(1);
    return nullptr;
  }
}

bool MLIRGenImpl::is_condition_expr(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "is_condition_expr";
  return ast.count("condition_expr");
}

mlir::Value
MLIRGenImpl::parse_condition_expr(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_condition_expr";
  auto lhs = parse_additive_expr(safe_get_child(get_item(ast, 0), "additive_expr"));
  for (size_t i = 1; i < ast.size(); i += 2) {
    auto op = safe_get_str(get_item(ast, i), "text");
    auto rhs = parse_additive_expr(safe_get_child(get_item(ast, i + 1), "additive_expr"));
    if (op == "==") {
      lhs = builder.create<mlir::arith::CmpIOp>(loc,mlir::arith::CmpIPredicate::eq, lhs, rhs);
    } else if (op == "!=") {
      lhs = builder.create<mlir::arith::CmpIOp>(loc,mlir::arith::CmpIPredicate::ne, lhs, rhs);
    }else if (op == "<=") {
      lhs = builder.create<mlir::arith::CmpIOp>(loc,mlir::arith::CmpIPredicate::sle, lhs, rhs);
    }else if (op == "<<") {
      lhs = builder.create<mlir::arith::CmpIOp>(loc,mlir::arith::CmpIPredicate::slt, lhs, rhs);
    }else if (op == ">=") {
      lhs = builder.create<mlir::arith::CmpIOp>(loc,mlir::arith::CmpIPredicate::sge, lhs, rhs);
    }else if (op == ">>") {
      lhs = builder.create<mlir::arith::CmpIOp>(loc,mlir::arith::CmpIPredicate::sgt, lhs, rhs);
    }else if (op == "&&") {
      lhs = builder.create<mlir::arith::AndIOp>(loc, lhs, rhs);
    }else{
      // raise: not support yet
      mlir::emitError(mlir::UnknownLoc::get(builder.getContext()),
                      "Not support condition op: " + op);
      std::exit(1);
      return nullptr;
    }
  }
  return lhs;
}

bool MLIRGenImpl::is_additive_expr(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "is_additive_expr";
  return ast.count("additive_expr");
}

mlir::Value
MLIRGenImpl::parse_additive_expr(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_additive_expr";
  auto lhs = parse_multiplicative_expr(safe_get_child(get_item(ast, 0), "multiplicative_expr"));
  for (size_t i = 1; i < ast.size(); i += 2) {
    auto op = safe_get_str(get_item(ast, i), "text");
    auto rhs = parse_multiplicative_expr(safe_get_child(get_item(ast, i + 1), "multiplicative_expr"));
    if (op == "+") {
      lhs = builder.create<mlir::arith::AddIOp>(loc, lhs, rhs);
    } else if (op == "-") {
      lhs = builder.create<mlir::arith::SubIOp>(loc, lhs, rhs);
    }else{
      // raise: not support yet
      mlir::emitError(mlir::UnknownLoc::get(builder.getContext()),
                      "Not support additive op: " + op);
      std::exit(1);
      return nullptr;
    }
  }
  return lhs;
}

mlir::Value
MLIRGenImpl::parse_multiplicative_expr(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_multiplicative_expr";
  auto lhs = parse_primary_expr(safe_get_child(get_item(ast, 0), "primary_expr"));
  for (size_t i = 1; i < ast.size(); i += 2) {
    auto op = safe_get_str(get_item(ast, i), "text");
    auto rhs = parse_primary_expr(safe_get_child(get_item(ast, i + 1), "primary_expr"));
    if (op == "*") {
      lhs = builder.create<mlir::arith::MulIOp>(loc, lhs, rhs);
    } else if (op == "/") {
      lhs = builder.create<mlir::arith::DivSIOp>(loc, lhs, rhs);
    } else if (op == "%") {
      lhs = builder.create<mlir::arith::RemSIOp>(loc, lhs, rhs);
    }
  }
  return lhs;
}

mlir::Value
MLIRGenImpl::parse_primary_expr(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_primary_expr";
  if (ast.size() >= 1 && get_item(ast, 0).count("unary_expr")) {
    auto primary_expr = get_item(ast, 0);
    return parse_unary_expr(safe_get_child(primary_expr, "unary_expr"));
  } else if (ast.size() >= 2 && get_item(ast, 1).count("expr")) {
    auto primary_expr = get_item(ast, 1);
    return parse_expr(safe_get_child(primary_expr, "expr"));
  } else {
    // raise: not support yet
    mlir::emitError(mlir::UnknownLoc::get(builder.getContext()),
                    "Not support primary_expr: " + ast.begin()->first);
    std::exit(1);
    return nullptr;
  }
}

mlir::SmallVector<mlir::Value>
MLIRGenImpl::parse_array_1d(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_array_1d";
  mlir::SmallVector<mlir::Value> values;
  for (const auto &pair : ast) {
    if (pair.second.count("text") &&
        (safe_get_str(pair.second, "text") == "[" or
         safe_get_str(pair.second, "text") == "]" or
         safe_get_str(pair.second, "text") == ",")) {
      continue;
    }
    values.push_back(parse_expr(safe_get_child(pair.second, "expr")));
  }
  LOG_DEBUG << "parse_array_1d finish";
  return values;
}

std::vector<int64_t>
MLIRGenImpl::parse_const_array1d(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_const_array1d";

  std::vector<int64_t> values;
  for (const auto &pair : ast) {
    if (pair.second.count("text") &&
        (safe_get_str(pair.second, "text") == "<" or
         safe_get_str(pair.second, "text") == ">" or
         safe_get_str(pair.second, "text") == ",")) {
      continue;
    }
    values.push_back(parse_const_int(safe_get_child(pair.second, "constant")));
  }
  return values;
}

std::vector<int64_t>
MLIRGenImpl::parse_shape(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_shape";

  std::vector<int64_t> shape =
      parse_const_array1d(safe_get_child(get_item(ast, 0), "const_array1d"));
  std::vector<int64_t> adjust_shape;
  for (auto it = shape.begin(); it != shape.end(); it++) {
    int64_t value = *it;
    if (value > 0) {
      adjust_shape.push_back(value);
    } else {
      adjust_shape.push_back(mlir::ShapedType::kDynamic);
    }
  }
  return adjust_shape;
  // return llvm::makeArrayRef(shape.begin(), shape.size());
}

mlir::Type MLIRGenImpl::parse_datatype(std::string datatype) {
  LOG_DEBUG << "parse_datatype";

  if (datatype == "int1") {
    return builder.getI1Type();
  } else if (datatype == "int8") {
    return builder.getI8Type();
  } else if (datatype == "int32") {
    return builder.getI32Type();
  } else if (datatype == "int64") {
    return builder.getI64Type();
  } else if (datatype == "index") {
    return builder.getIndexType();
  } else if (datatype == "float32") {
    return builder.getF32Type();
  } else if (datatype == "fp16") {
    return builder.getF16Type();
  } else {
    // raise: not support yet
    mlir::emitError(mlir::UnknownLoc::get(builder.getContext()),
                    "Not support datatype: " + datatype);
    std::exit(1);
    return nullptr;
  }
}

std::string MLIRGenImpl::parse_memory_type(std::string memory_type) {
  std::string result = memory_type;

  // 去掉前缀下划线
  size_t start = result.find_first_not_of('_');
  if (start != std::string::npos) {
    result.erase(0, start);
  }

  // 去掉后缀下划线
  size_t end = result.find_last_not_of('_');
  if (end != std::string::npos) {
    result.erase(end + 1);
  }

  // 将所有字母转为小写
  for (char &ch : result) {
    ch = std::tolower(ch);
  }
  LOG_DEBUG << "Convert " << memory_type << " to " << result;
  return result;
}

mlir::Attribute MLIRGenImpl::parse_device(std::string device) {
  LOG_DEBUG << "parse_device";

  mlir::SmallVector<mlir::NamedAttribute, 1> nameAttrs;
  nameAttrs.push_back(
      builder.getNamedAttr("address", builder.getI64IntegerAttr(-1)));
  mlir::DictionaryAttr attr =
      mlir::DictionaryAttr::get(builder.getContext(), nameAttrs);
  return attr;
}

mlir::MemRefType
MLIRGenImpl::parse_param_type_tensor(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_param_type_tensor";

  // shape
  auto param_type_shape = safe_get_child(get_item(ast, 1), "param_type_shape");
  auto shape = parse_shape(param_type_shape);

  // datatype
  auto datatype = parse_datatype(safe_get_str(get_item(ast, 3), "text"));

  // device
  auto device = parse_device(safe_get_str(get_item(ast, 5), "text"));

  std::vector<int64_t> unknown_strides(shape.size(), mlir::ShapedType::kDynamic);
  mlir::MemRefType type =
      mlir::MemRefType::get(llvm::ArrayRef<int64_t>(shape), 
                            datatype,
                            mlir::StridedLayoutAttr::get(
                              builder.getContext(), 
                              mlir::ShapedType::kDynamic,  // unknown offset
                              llvm::ArrayRef<int64_t>(unknown_strides)  // unknown strides
                            ),
                            device);

  return type;
}

mlir::UnrankedMemRefType
MLIRGenImpl::parse_param_type_unranked_tensor(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_param_type_unranked_tensor";

  // shape
  // auto param_type_shape = safe_get_child(get_item(ast, 1), "param_type_shape");
  // auto shape = parse_shape(param_type_shape);

  // datatype
  auto datatype = parse_datatype(safe_get_str(get_item(ast, 1), "text"));

  // device
  auto device = parse_device(safe_get_str(get_item(ast, 3), "text"));

  // build the type
  mlir::UnrankedMemRefType type =
      mlir::UnrankedMemRefType::get(datatype, device);

  return type;
}

mlir::Type
MLIRGenImpl::parse_param_type_scalar(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_param_type_scalar";

  // datatype
  auto datatype = parse_datatype(safe_get_str(get_item(ast, 1), "text"));

  return datatype;
}

mlir::Type
MLIRGenImpl::parse_param_type(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_param_type";
  auto ast_param_type = get_item(ast, 0);
  if (ast_param_type.count("param_type_tensor")) {
    return parse_param_type_tensor(
        safe_get_child(ast_param_type, "param_type_tensor"));
  } else if (ast_param_type.count("param_type_unranked_tensor")) {
    return parse_param_type_unranked_tensor(
        safe_get_child(ast_param_type, "param_type_unranked_tensor"));
  } else if (ast_param_type.count("param_type_scalar")) {
    return parse_param_type_scalar(
        safe_get_child(ast_param_type, "param_type_scalar"));
  } else {
    // raise: not support yet
    mlir::emitError(mlir::UnknownLoc::get(builder.getContext()),
                    "Not support param_type: " + ast_param_type.begin()->first);
    std::exit(1);
    return nullptr;
  }
}

std::pair<mlir::Type, std::string>
MLIRGenImpl::parse_param_type_and_name(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_tensor_type_and_name";

  auto args_name = safe_get_child(get_item(ast, 0), "param_name");
  std::string name = safe_get_str(get_item(args_name, 0), "text");

  mlir::Type args_type =
      parse_param_type(safe_get_child(get_item(ast, 1), "param_type"));
  return std::make_pair(args_type, name);
}

std::pair<std::vector<mlir::Type>, std::vector<std::string>>
MLIRGenImpl::parse_func_args(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "parse_func_args";

  std::vector<mlir::Type> args_types;
  std::vector<std::string> args_names;
  int i = 0;
  for (const auto &pair : ast) {
    if (is_tensor_args(pair.second) || is_scalar_args(pair.second)) {
      auto type_and_name =
          parse_param_type_and_name(safe_get_child(pair.second, "func_param"));
      args_types.push_back(type_and_name.first);
      args_names.push_back(type_and_name.second);
    }
  }
  return std::make_pair(args_types, args_names);
}

bool MLIRGenImpl::is_tensor_args(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "is_tensor_args";

  if (!ast.count("func_param"))
    return false;
  auto func_param = safe_get_child(ast, "func_param");
  auto param_type = safe_get_child(get_item(func_param, 1), "param_type");
  return get_item(param_type, 0).count("param_type_tensor") || get_item(param_type, 0).count("param_type_unranked_tensor");
}

bool MLIRGenImpl::is_scalar_args(const boost::property_tree::ptree &ast) {
  LOG_DEBUG << "is_scalar_args";

  if (!ast.count("func_param"))
    return false;
  auto func_param = safe_get_child(ast, "func_param");
  auto param_type = safe_get_child(get_item(func_param, 1), "param_type");
  return get_item(param_type, 0).count("param_type_scalar");
}

mlir::SmallVector<mlir::Value>
MLIRGenImpl::cast_to_index_type(mlir::SmallVector<mlir::Value> _index) {
  // cast to index
  // mlir::SmallVector<mlir::Value> index;
  // for(int i = 0; i<_index.size(); i++){
  //     mlir::Value index_i = builder.create<mlir::index::CastSOp>(loc,
  //     builder.getIndexType(), _index[i]); index.push_back(index_i);
  // }
  return _index;
}

// std::vector<mlir::scf::ForOp> MLIRGenImpl::getUnrollForOps() {
//   return unrollForOps;
// }