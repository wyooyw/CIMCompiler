# CIMCompiler

CIMCompiler 是一款专为 SRAM 存内计算架构（Compute-In-Memory, CIM）设计的张量编译器，其核心是基于MLIR开发而成。它能够将基于 CIM-DSL（领域特定语言）编写的张量计算程序编译成 CIM-ISA 汇编代码，实现在 CIM 模拟器上的高效执行。作为 CIMFlow 生态系统的核心组件之一，CIMCompiler 致力于推动 SRAM 存内计算架构的技术创新与实际应用。

## CIMCompiler组成部分

### 编译器

使用Antlr实现词法分析和语法分析，随后转为MLIR的中间表示。

我们复用了MLIR中的一些Dialect，并额外实现了CIM-Dialect和CIM-ISA-Dialect来进行扩展。

我们复用了MLIR中的一些通用的Pass，并额外实现了一些Pass来完成一些特殊的需求。

我们暂时编写了一套简单的代码生成模块，用于生成最终的CIM-ISA汇编程序。后续计划将该部分下沉到LLVM中进行更完整的实现。

### 算子库

CIM-DSL简洁且灵活，能够用其快速编写各种算子，目前已经支持的有：
- 常见的神经网络算子：2d卷积、深度可分离卷积、池化、全连接层等
- 稀疏2d卷积算子：值稀疏、比特级稀疏和值-比特混合稀疏实现

### 行为级模拟器

该模拟器为CIMCompiler中自带的模拟器，是一个简单的行为级模拟器。该模拟器仅用于快速验证编译指令的正确性，并不能像CIM-Simulator一样评估性能。

## 相关资源

- **CIMFlow 项目**：了解 SRAM 存内计算架构的完整技术细节和 CIM-ISA 指令集架构
- **CIM-Simulator**：配套的存内计算架构模拟器
- **技术论文**：获取相关学术研究成果和技术原理

## 快速开始

### 安装依赖

TODO

### 编译

```
pip install -e .
```

### 运行

使用CLI

```
cim-compiler compile \
-i test/compiler/control_flow/if/if/code.cim \
-o ./temp \
-c test/compiler/config.json
```

### 测试

```
CIM_COMPILER_HOME=. PYTHONPATH=. pytest test
```