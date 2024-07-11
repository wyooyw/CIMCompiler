grammar CIM;               // 定义文法的名字

program: define_function+;

// Function
define_function:'def' ID '(' func_param_list ')' '{' func_body '}';
func_param_list: func_param (',' func_param)*;
func_body: stmt_list;

// 
func_param: param_name param_type;
param_name: ID;
param_type: param_type_tensor | param_type_scalar;
param_type_tensor: '<' param_type_shape ',' DATATYPE ',' MEMORY '>';
param_type_shape: const_array1d;

param_type_scalar: '<' DATATYPE '>';

// Statement
stmt_list: (stmt )+;
stmt: (
    stmt_assign
    | stmt_call
    | stmt_for
    | stmt_return
    ) ';' ;
stmt_assign: ID EQ expr;
stmt_call: call;
stmt_for: 'for' ID 'in' for_range carry '{' stmt_list '}';

for_range : for_range_1 | for_range_2 | for_range_3;
for_range_1: 'range(' const_or_var ')';
for_range_2: 'range(' const_or_var ',' const_or_var')';
for_range_3: 'range(' const_or_var ',' const_or_var ',' const_or_var ')';

carry: 'carry' '(' carry_list ')';
carry_list: var (',' var)*;

stmt_return: 'return' ID;



// Expression
expr: 
    unary_expr
    | binary_expr;
unary_expr: 
    call
    | const_or_var;
binary_expr: unary_expr BINARY_OP unary_expr;

// Call
call: ID '(' call_param_list ')';
call_param_list: call_param (',' call_param)*;
call_param: datatype | memory | const_array1d | array1d | expr;
datatype: DATATYPE;
memory: MEMORY;

// Term
const_or_var: constant | var;
constant: CONST;
var: ID;
const_array1d: '<' constant (',' constant)* '>';
array1d: '[' expr (',' expr)* ']';

MEMORY: ('global' | 'local' | 'macro' | 'rf') ;
DATATYPE: ('int8' | 'int32' | 'int64' | 'index' | 'float32') ;
BINARY_OP: ADD | SUB | MUL | DIV | MOD;
ADD : '+';
SUB : '-';
MUL : '*';
DIV : '/';
MOD : '%';
CONST: ( CONST_NEG | CONST_POS );
CONST_NEG : '-' CONST_POS;
CONST_POS : [0-9]+;
EQ : '=';
ID : [a-zA-Z_]+[a-zA-Z_0-9]* ;               // 标志符由大小写字母,下划线和数字组成。数字不能开头
WS : [ \t\r\n]+ -> skip ;    // 跳过空格、制表符、回车符和换行符

