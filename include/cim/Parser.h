#ifndef MLIRGENIMPL_H
#define MLIRGENIMPL_H
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/InitAllDialects.h"
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <iostream>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

class MLIRGenImpl {
public:
  explicit MLIRGenImpl(mlir::MLIRContext &context);

  mlir::ModuleOp parseJson(std::string json_path);
  mlir::ModuleOp parseModule(const boost::property_tree::ptree &ast);
  std::vector<mlir::scf::ForOp> getUnrollForOps();

private:
  mlir::OpBuilder builder;
  mlir::Location loc;
  mlir::StringAttr GLOBAL_MEMORY;
  mlir::StringAttr LOCAL_MEMORY;
  mlir::StringAttr MACRO_MEMORY;
  std::unordered_map<std::string, std::unordered_map<std::string, mlir::Value>>
      signature_table;
  std::unordered_map<std::string, mlir::func::FuncOp> signature_table_func;
  std::string current_func_name;
  std::stack<mlir::Block *> block_stack;
  std::vector<mlir::scf::ForOp> unrollForOps;

  const boost::property_tree::ptree &
  safe_get_child(const boost::property_tree::ptree &ast,
                 const std::string &key);
  std::string safe_get_str(const boost::property_tree::ptree &ast,
                           const std::string &key);

  void init_func_in_sign_table(const std::string &func_name);
  void add_to_sign_table(const std::string &arg_name, mlir::Value arg);
  mlir::Value get_from_sign_table(const std::string &arg_name);
  void add_func_to_sign_table(const std::string &func_name,
                              mlir::func::FuncOp func);
  mlir::func::FuncOp get_func_from_sign_table(const std::string &func_name);
  void parse_func(const boost::property_tree::ptree &ast);
  void parse_func_body(const boost::property_tree::ptree &ast);
  void parse_stmt_list(const boost::property_tree::ptree &ast);
  void parse_stmt(const boost::property_tree::ptree &ast);
  bool is_assign_stmt(const boost::property_tree::ptree &ast);
  bool is_return_stmt(const boost::property_tree::ptree &ast);
  bool is_call_stmt(const boost::property_tree::ptree &ast);
  bool is_for_stmt(const boost::property_tree::ptree &ast);
  bool is_if_else_stmt(const boost::property_tree::ptree &ast);
  void parse_assign_stmt(const boost::property_tree::ptree &ast);
  void parse_for_stmt(const boost::property_tree::ptree &ast);
  void parse_if_else_stmt(const boost::property_tree::ptree &ast);
  mlir::Value parse_expr(const boost::property_tree::ptree &ast);
  bool is_unary_expr(const boost::property_tree::ptree &ast);
  bool is_binary_expr(const boost::property_tree::ptree &ast);
  mlir::Value parse_unary_expr(const boost::property_tree::ptree &ast);
  mlir::Value parse_binary_expr(const boost::property_tree::ptree &ast);
  mlir::Value parse_var(const boost::property_tree::ptree &ast);
  mlir::Value parse_const(const boost::property_tree::ptree &ast);
  bool is_const(const boost::property_tree::ptree &ast);
  bool is_var(const boost::property_tree::ptree &ast);
  mlir::Value parse_const_or_var(const boost::property_tree::ptree &ast);
  bool is_const_or_var(const boost::property_tree::ptree &ast);
  bool is_call(const boost::property_tree::ptree &ast);
  // mlir::Value parse_unary_expr_scalar(const boost::property_tree::ptree&
  // ast); mlir::Value parse_binary_expr_scalar(const
  // boost::property_tree::ptree& ast); bool is_unary_expr_scalar(const
  // boost::property_tree::ptree& ast); bool is_binary_expr_scalar(const
  // boost::property_tree::ptree& ast); mlir::Value parse_expr_scalar(const
  // boost::property_tree::ptree& ast);
  mlir::SmallVector<mlir::Value>
  parse_array_1d(const boost::property_tree::ptree &ast);
  std::vector<int64_t> parse_shape(const boost::property_tree::ptree &ast);
  mlir::Type parse_datatype(std::string datatype);
  mlir::Attribute parse_device(std::string device);
  mlir::MemRefType
  parse_param_type_tensor(const boost::property_tree::ptree &ast);
  mlir::Type parse_param_type_scalar(const boost::property_tree::ptree &ast);
  mlir::Type parse_param_type(const boost::property_tree::ptree &ast);
  std::pair<mlir::Type, std::string>
  parse_param_type_and_name(const boost::property_tree::ptree &ast);
  std::pair<std::vector<mlir::Type>, std::vector<std::string>>
  parse_func_args(const boost::property_tree::ptree &ast);
  bool is_tensor_args(const boost::property_tree::ptree &ast);
  bool is_scalar_args(const boost::property_tree::ptree &ast);

  std::vector<int64_t>
  parse_const_array1d(const boost::property_tree::ptree &ast);
  int64_t parse_const_int(const boost::property_tree::ptree &ast);

  mlir::Value build_shape_op(mlir::ValueRange param_list);

  // Range
  std::vector<mlir::Value>
  parse_for_range(const boost::property_tree::ptree &ast);
  std::vector<mlir::Value>
  parse_for_range_1(const boost::property_tree::ptree &ast);
  std::vector<mlir::Value>
  parse_for_range_2(const boost::property_tree::ptree &ast);
  std::vector<mlir::Value>
  parse_for_range_3(const boost::property_tree::ptree &ast);

  // Carry
  std::pair<llvm::SmallVector<std::string>, llvm::SmallVector<mlir::Value>>
  parse_carry(const boost::property_tree::ptree &ast);
  std::pair<llvm::SmallVector<std::string>, llvm::SmallVector<mlir::Value>>
  parse_carry_list(const boost::property_tree::ptree &ast);
  std::pair<std::string, mlir::Value>
  parse_var_and_name(const boost::property_tree::ptree &ast);

  // Call
  void parse_call_stmt(const boost::property_tree::ptree &ast);
  void parse_call(const boost::property_tree::ptree &ast);
  mlir::Value parse_call_return_value(const boost::property_tree::ptree &ast);
  llvm::SmallVector<mlir::Value>
  parse_call_param_list(const boost::property_tree::ptree &ast);
  mlir::Value parse_call_param(const boost::property_tree::ptree &ast);

  // Builtin Functions
  mlir::Value parse_builtin_shape(const boost::property_tree::ptree &ast);
  void parse_builtin_trans(const boost::property_tree::ptree &ast);
  mlir::Value parse_builtin_slice(const boost::property_tree::ptree &ast);
  void parse_builtin_vvadd(const boost::property_tree::ptree &ast);
  void parse_builtin_vvmul(const boost::property_tree::ptree &ast);
  void parse_builtin_quantify(const boost::property_tree::ptree &ast);
  mlir::Value parse_builtin_buffer(const boost::property_tree::ptree &ast);
  void parse_builtin_print(const boost::property_tree::ptree &ast);
  void parse_builtin_debug(const boost::property_tree::ptree &ast);
  void parse_builtin_free(const boost::property_tree::ptree &ast);
  mlir::Value parse_builtin_load(const boost::property_tree::ptree &ast);
  mlir::Value parse_builtin_min(const boost::property_tree::ptree &ast);
  mlir::Value parse_builtin_max(const boost::property_tree::ptree &ast);
  mlir::Value parse_builtin_addr(const boost::property_tree::ptree &ast);
  mlir::Value parse_builtin_select(const boost::property_tree::ptree &ast);
  void parse_builtin_save(const boost::property_tree::ptree &ast);
  void parse_builtin_cimcompute_dense(const boost::property_tree::ptree &ast);
  void
  parse_builtin_cimcompute_value_sparse(const boost::property_tree::ptree &ast);
  void
  parse_builtin_cimcompute_bit_sparse(const boost::property_tree::ptree &ast);
  void parse_builtin_cimcompute_value_bit_sparse(
      const boost::property_tree::ptree &ast);
  void parse_builtin_cimcompute(const boost::property_tree::ptree &ast,
                                bool value_sparse, bool bit_sparse);
  void parse_builtin_cimoutput(const boost::property_tree::ptree &ast);
  void parse_builtin_cimoutput_sum(const boost::property_tree::ptree &ast);
  void parse_builtin_cimtransfer(const boost::property_tree::ptree &ast);
  void parse_builtin_cimset(const boost::property_tree::ptree &ast);
  void parse_builtin_special_reg_set(const boost::property_tree::ptree &ast);
  // mlir::Value parse_builtin_reshape(const boost::property_tree::ptree& ast);

  // Util Functions
  mlir::SmallVector<mlir::Value>
  cast_to_index_type(mlir::SmallVector<mlir::Value> _index);
};

#endif // MLIRGENIMPL_H