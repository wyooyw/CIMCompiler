import sqlite3
from dataclasses import dataclass


@dataclass
class Experiment:
    time: int
    message: str
    id: int = None


@dataclass
class ExperimentResult:
    experiment_id: int
    op: str
    strategy: str
    status: int
    save_path: str
    macro_ultilization: float
    macro_compute_times: int
    flops: int
    need_minimize_macros: int
    latency: int
    energy: int
    config: str
    time: int
    id: int = None


class DataBase:
    def __init__(self, db_name):
        self.db_name = db_name
        self._create_and_connect_db()
        self._create_table()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _create_and_connect_db(self):
        self.conn = sqlite3.connect(self.db_name)
        # 设置 row_factory 为 sqlite3.Row，使结果支持通过列名访问
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()

    def _create_table(self):
        # 实验表
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS experiment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time INTEGER,
                message TEXT
            )
        """
        )
        self.conn.commit()

        # 实验结果表
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS exp_result (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                op TEXT,
                strategy TEXT,
                status INTEGER,
                save_path TEXT,
                macro_ultilization REAL,
                macro_compute_times INTEGER,
                flops INTEGER,
                need_minimize_macros INTEGER,
                latency REAL,
                energy REAL,
                config TEXT,
                time INTEGER,
                FOREIGN KEY (experiment_id) REFERENCES experiment(id)
            )
        """
        )
        self.conn.commit()

    def insert_experiment(self, experiment: Experiment) -> int:
        # 插入实验数据
        self.cursor.execute(
            "INSERT INTO experiment (time, message) VALUES (?, ?)",
            (experiment.time, experiment.message),
        )
        self.conn.commit()
        # 返回最后插入行的id
        return self.cursor.lastrowid

    def insert_experiment_result(self, exp_result: ExperimentResult) -> int:
        # 插入实验结果数据
        self.cursor.execute(
            "INSERT INTO exp_result (experiment_id, op, strategy, status, save_path, macro_ultilization, macro_compute_times, flops, need_minimize_macros, latency, energy, config, time) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                exp_result.experiment_id,
                exp_result.op,
                exp_result.strategy,
                exp_result.status,
                exp_result.save_path,
                exp_result.macro_ultilization,
                exp_result.macro_compute_times,
                exp_result.flops,
                exp_result.need_minimize_macros,
                exp_result.latency,
                exp_result.energy,
                exp_result.config,
                exp_result.time,
            ),
        )

        self.conn.commit()
        # 返回最后插入行的id
        return self.cursor.lastrowid

    def get_all_experiments(self) -> list[Experiment]:
        self.cursor.execute("SELECT * FROM experiment ORDER BY time DESC")
        rows = self.cursor.fetchall()
        # Convert tuples to dictionaries with column names
        return [Experiment(id=row[0], time=row[1], message=row[2]) for row in rows]

    def get_experiment_results_by_experiment_id(
        self, experiment_id: int
    ) -> list[ExperimentResult]:
        self.cursor.execute(
            "SELECT * FROM exp_result WHERE experiment_id = ?", (experiment_id,)
        )
        rows = self.cursor.fetchall()
        return [
            ExperimentResult(
                id=row["id"],
                experiment_id=row["experiment_id"],
                op=row["op"],
                strategy=row["strategy"],
                status=row["status"],
                save_path=row["save_path"],
                macro_ultilization=row["macro_ultilization"],
                macro_compute_times=row["macro_compute_times"],
                flops=row["flops"],
                need_minimize_macros=row["need_minimize_macros"],
                latency=row["latency"],
                energy=row["energy"],
                config=row["config"],
                time=row["time"],
            )
            for row in rows
        ]

    def close(self):
        self.conn.close()


# if __name__=="__main__":
#     with DataBase("iccad25.db") as db:
#         print(db.get_all_experiments())
