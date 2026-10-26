# A 股 XGBoost 量化研究工具

一个基于 **Python + Tkinter + XGBoost** 的 A 股量化模型训练与研究工具。项目提供图形化界面，用于完成行情数据获取、特征工程、模型训练、策略回测、参数寻优与自主研究流程。

> 本项目面向技术研究与开源交流，不提供任何投资建议。

---

## 核心亮点

- **141 个自动化测试保障**  
  覆盖数据读取、标签构造、训练流程、滚动验证、晋级判断、SQLite 登记、异常恢复等关键路径。

- **SQLite 数据库防测试集泄漏锁**  
  手工训练与自主研究共用测试集使用登记。测试窗口一旦被领取，重叠区间会被阻止，避免反复使用同一段测试数据造成评估污染。

- **极致简单的跨平台 GUI 启动**  
  无需 Web 服务、无需前端构建。安装依赖后直接运行 `train_xgb_gui.py` 即可启动桌面界面。

- **严格的时间序列纪律**  
  训练集、验证集、测试集按时间切分；标签使用未来收益构造，但交易执行遵循 T+1，避免未来函数。

- **面向普通电脑的预算控制**  
  限制参数试验次数、特征组合数量与线程数，适合 4GB 内存级别的老电脑运行。

---

## 功能概览

- 股票代码与训练/测试区间配置
- AkShare 行情数据获取与本地缓存
- 技术指标特征工程（MA、RSI、MACD、KDJ、BOLL、成交量、ATR 等）
- XGBoost 多分类训练与早停
- 策略参数寻优（止损、止盈、均线周期）
- 特征组合寻优
- 批量跑测与历史记录
- 资金曲线与回测结果展示
- 自主研究调度：选股 → 数据检查 → 训练 → 验证 → 晋级/拒绝
- 研究队列与实验记录（SQLite）
- 一键重启、自动备份与异常诊断日志

---

## 环境要求

- **Python 3.10+**
- **Tkinter**（Python 标准库；macOS 建议使用官方 Python 安装包，已适配 Tk 8.6）
- 支持平台：
  - **macOS**（已在 Apple Silicon / Tk 8.6 环境验证）
  - **Windows**（保留 `run_train_gui.bat` 启动方式）
- 推荐使用项目内虚拟环境 `.venv`，不要污染全局 Python。

---

## 快速开始

### 1. 克隆项目

```bash
git clone <your-repository-url>
cd <project-directory>
```

### 2. 创建虚拟环境

macOS / Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows：

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. 启动 GUI

macOS / Linux：

```bash
python train_xgb_gui.py
```

Windows：

```bat
run_train_gui.bat
```

或直接：

```bat
python train_xgb_gui.py
```

---

## 运行测试

项目使用 Python 标准库 `unittest`，无需额外安装测试框架。

```bash
python -m unittest discover tests -v
```

预期结果：

```text
Ran 141 tests
OK
```

> 部分真实行情测试依赖网络；网络不可用时可能被跳过，这是预期行为。

---

## 项目结构

```text
.
├── train_xgb_gui.py          # GUI 入口：训练、回测、历史记录、自主研究控制
├── auto_research.py          # 自动研究执行器：实验流水线、训练与报告
├── research_engine.py        # 自主研究调度、晋级规则、滚动验证
├── stock_pool.py             # 股票池获取、数据检查、研究队列（SQLite）
├── memory_guard.py           # 内存保护与重型任务闸门
├── safe_io.py                # 原子读写工具
├── model_params.json         # 模型参数模板
├── memory_rules.json         # 内存保护规则
├── promotion_rules.json      # 晋级验收规则
├── research_engine_config.json
├── research_scope.json
├── requirements.txt
├── run_train_gui.bat         # Windows 启动脚本
├── experiments/              # 实验产物目录（运行时生成，默认不提交）
└── tests/                    # 141 个自动化测试
```

运行时会自动生成以下本地数据（已加入 `.gitignore`，不会提交到仓库）：

- `experiments/`：每次实验的模型、报告与摘要
- `equity_curves/`：资金曲线
- `backups/`：程序自动备份
- `history_*.csv`、`tuning_log_*.csv`、`feature_search_log_*.csv`
- `model_*.json`
- `research_queue.db`
- `*.log`、`*.bak`、临时诊断文件

---

## 数据与安全说明

- 本项目不会上传或同步你的本地行情数据、模型文件与实验记录。
- 所有 API 访问均通过 AkShare / requests 的默认 HTTPS 证书校验，不包含关闭证书校验的逻辑。
- 代码中不包含硬编码的 API Key、密码、访问令牌或代理节点配置。
- 训练与回测均在本地执行，不会向外部服务发送你的实验数据。

---

## 免责声明

**本项目仅供技术研究与开源交流，不构成任何投资建议。**

- 历史回测收益不代表未来真实收益。
- 模型训练结果受数据质量、市场环境、参数选择等多种因素影响。
- 股市有风险，投资需谨慎。因使用本项目产生的任何直接或间接损失，作者不承担责任。
- 请遵守所在地区法律法规及数据提供方的使用条款。

---

## 贡献

欢迎提交 Issue、改进建议或 Pull Request。提交前请确保：

1. 不包含真实账户、密钥、代理配置或个人数据；
2. 不提交本地模型、行情数据与实验产物；
3. 运行 `python -m unittest discover tests -v`，确保 141 个测试全部通过。
