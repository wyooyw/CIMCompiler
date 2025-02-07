import os
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Any

class StatsProcessor:
    def __init__(self, stats_schema: Dict[str, List[str]]):
        """
        初始化处理器
        
        Args:
            stats_schema: 统计数据的schema定义
            例如: {
                "scalar": ["total", "general_li", "special_li"],
                "trans": ["total"],
                "pim": ["total", "pim_set", "pim_compute"],
                "simd": ["total", "quantify"]
            }
        """
        self.stats_schema = stats_schema
        self.column_names = self._generate_column_names()
        self.create_table_sql = self._generate_create_table_sql()
        self.insert_sql = self._generate_insert_sql()

    def _generate_column_names(self) -> List[str]:
        """生成所有列名"""
        columns = ["model_name", "mode", "layer_name", "total"]
        for category, fields in self.stats_schema.items():
            for field in fields:
                columns.append(f"{category}_{field}")
        return columns

    def _generate_create_table_sql(self) -> str:
        """生成建表SQL语句"""
        columns = [f"{col} TEXT" for col in self.column_names]
        return f"CREATE TABLE IF NOT EXISTS stats ({', '.join(columns)})"

    def _generate_insert_sql(self) -> str:
        """生成插入数据SQL语句"""
        placeholders = ','.join(['?' for _ in self.column_names])
        return f"INSERT INTO stats ({','.join(self.column_names)}) VALUES ({placeholders})"

    def _extract_stats_values(self, stats: Dict[str, Any]) -> List[Any]:
        """从stats数据中提取值"""
        values = []
        values.append(stats.get('total'))
        for category, fields in self.stats_schema.items():
            category_data = stats.get(category, {})
            for field in fields:
                values.append(category_data.get(field, 0))
        return values

    def process_stats_files(self, base_path: str):
        """处理统计文件"""
        db_path = os.path.join(base_path, 'stats.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 创建表
        cursor.execute(self.create_table_sql)

        # 遍历所有stats.json文件
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file == 'stats.json':
                    file_path = os.path.join(root, file)
                    
                    # 解析路径获取信息
                    rel_path = os.path.relpath(file_path, base_path)
                    parts = Path(rel_path).parts
                    if len(parts) >= 4:
                        model_name = parts[1]
                        mode = parts[2]
                        layer_name = parts[3]

                        # 读取JSON文件
                        with open(file_path, 'r') as f:
                            stats = json.load(f)

                        # 准备数据
                        values = [model_name, mode, layer_name]
                        values.extend(self._extract_stats_values(stats))

                        # 插入数据
                        cursor.execute(self.insert_sql, values)

        # 提交更改并关闭连接
        conn.commit()
        conn.close()

import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
from typing import Dict, List

class StatsVisualizer:
    def __init__(self, db_path: str, stats_schema: Dict[str, List[str]] = None):
        self.db_path = db_path
        self.stats_schema = stats_schema
        self.columns = self._get_columns()

    def _get_columns(self) -> List[str]:
        """获取数据表的列名"""
        if self.stats_schema:
            # 方法1：从stats_schema生成列名
            columns = ["model_name", "mode", "layer_name", "total"]
            for category, fields in self.stats_schema.items():
                for field in fields:
                    columns.append(f"{category}_{field}")
        else:
            # 方法2：从数据库表直接获取列名
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM stats LIMIT 1")
            columns = [description[0] for description in cursor.description]
            conn.close()
        return columns

    def _generate_aggregate_query(self) -> str:
        """生成聚合查询SQL"""
        # 排除layer_name列（因为要按model_name和mode聚合）
        agg_columns = [col for col in self.columns if col != 'layer_name']
        
        # 为每个数值列生成SUM表达式
        sum_expressions = []
        for col in agg_columns:
            if col in ['model_name', 'mode']:
                sum_expressions.append(col)
            else:
                sum_expressions.append(f"SUM(CAST({col} AS INTEGER)) as {col}")
        
        query = f"""
        SELECT {', '.join(sum_expressions)}
        FROM stats
        GROUP BY model_name, mode
        """
        return query

    def load_data(self) -> pd.DataFrame:
        """从数据库加载数据"""
        conn = sqlite3.connect(self.db_path)
        query = self._generate_aggregate_query()
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

    def visualize(self):
        """创建可视化界面"""
        st.title("模型统计数据可视化")
        
        # 加载数据
        df = self.load_data()
        
        # 侧边栏：选择要显示的模型
        models = df['model_name'].unique()
        selected_models = st.sidebar.multiselect(
            "选择模型",
            models,
            default=models
        )
        
        # 过滤数据
        filtered_df = df[df['model_name'].isin(selected_models)]
        
        # 展示数据表格
        st.subheader("聚合数据表格")
        st.dataframe(filtered_df)
        
        # 创建柱状图
        st.subheader("各模型在不同模式下的总指令数")
        fig = px.bar(
            filtered_df,
            x='model_name',
            y='total',
            color='mode',
            barmode='group',
            title='模型总指令数比较'
        )
        st.plotly_chart(fig)
        
        # 如果存在stats_schema，使用它来组织指令类型的展示
        if self.stats_schema:
            # 展示各类指令的占比
            st.subheader("指令类型分布")
            instruction_types = [f"{category}_total" for category in self.stats_schema.keys()]
            
            for model in selected_models:
                model_data = filtered_df[filtered_df['model_name'] == model]
                
                for mode in model_data['mode'].unique():
                    mode_data = model_data[model_data['mode'] == mode]
                    
                    # 创建饼图
                    pie_data = {
                        'type': instruction_types,
                        'count': [mode_data[t].iloc[0] for t in instruction_types]
                    }
                    pie_df = pd.DataFrame(pie_data)
                    
                    fig = px.pie(
                        pie_df,
                        values='count',
                        names='type',
                        title=f'{model} - {mode} 指令类型分布'
                    )
                    st.plotly_chart(fig)
            
            # 对每个类别的详细分析
            for category, fields in self.stats_schema.items():
                if len(fields) > 1:  # 如果该类别有多个字段
                    st.subheader(f"{category}指令分析")
                    category_fields = [f"{category}_{field}" for field in fields]
                    fig = px.bar(
                        filtered_df,
                        x='model_name',
                        y=category_fields,
                        color='mode',
                        barmode='group',
                        title=f'{category}指令详细分布'
                    )
                    st.plotly_chart(fig)

def main():
    # 定义统计数据的schema
    stats_schema = {
        "scalar": ["total", "general_li", "special_li"],
        "trans": ["total"],
        "pim": ["total", "pim_set", "pim_compute"],
        "simd": ["total", "quantify"]
    }
    
    # 创建可视化器并运行
    visualizer = StatsVisualizer(".result/2024-12-14/stats.db", stats_schema)
    visualizer.visualize()

if __name__ == "__main__":
    main()

# 使用示例
# if __name__ == "__main__":
#     # 定义统计数据的schema
#     stats_schema = {
#         "scalar": ["total", "general_li", "special_li"],
#         "trans": ["total"],
#         "pim": ["total", "pim_set", "pim_compute"],
#         "simd": ["total", "quantify"]
#     }

#     # 创建处理器并处理文件
#     processor = StatsProcessor(stats_schema)
#     processor.process_stats_files(".result/2024-12-14")