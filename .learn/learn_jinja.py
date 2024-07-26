from jinja2 import Environment, FileSystemLoader
import os
# 创建 Jinja2 环境和加载器
template_path = os.path.dirname(os.path.abspath(__file__))
env = Environment(loader=FileSystemLoader(template_path))

# 加载模板
template = env.get_template('template.jinja')

# 准备模板上下文
context = {
    'name': 'Alice',
    'age': 20,
    'items': ['Apple', 'Banana', 'Cherry']
}

# 渲染模板
output = template.render(context)

# 打印输出结果
print(output)