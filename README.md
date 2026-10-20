# Soccer Analytics

研究足球比赛、球队与球员，并用有时间边界的实验比较模型。范围包括男足五大联赛、欧冠、欧洲杯、世界杯和 2025 世俱杯。赛果、事件、有限站位快照、连续追踪、官方报告和游戏评分分别解释，不能互相替代。

[数据来源与存储](DATA_SOURCES.md) · [研究结果](results/README.md) · [复现命令](REPRODUCING.md) · [实现与验证记录](IMPLEMENTATION_REPORT.md)

## 浏览研究页面

仓库已包含经过筛选的公开汇总，浏览页面不需要完整数据库。

```bash
python3 -m http.server 8000 --bind 127.0.0.1 --directory docs
```

打开 `http://localhost:8000/`。页面通过 HTTP 读取 JSON，直接双击 HTML 文件不能完成全部检查。

| 入口 | 内容 |
| --- | --- |
| [概览](docs/index.html) | 公开样本、研究入口和实际快照日期 |
| [比赛](docs/matches.html) | 2024 欧洲杯事件、FIFA 接应报告、决赛案例 |
| [球队与球员](docs/entities.html) | 同来源、同联赛、同赛季比较，以及球风表示实验 |
| [模型实验](docs/models.html) | 比分、评级、树搜索与校准的时间留出结果，以及事件表示和 xT 诊断 |
| [数据与资料](docs/library.html) | 图表、覆盖、机构方法与论文 |
| [预测与发布检查](docs/predictions.html) | 独立的预测快照、历史验证、市场参考与发布门槛 |

筛选链接可恢复当前视图，数值表支持显示精度下的 CSV/JSON 导出，图表支持放大与完整 SVG 下载。旧 `research.html#…` 链接保留兼容。不同来源和年份的筛选分别生效，研究覆盖日期不会替代预测快照日期。

## 环境与命令

使用 Python 3.12。基础预测依赖、研究模型、图表、额外树引擎、深度学习和来源包装分别安装。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m app.research --help
```

| 依赖文件 | 用途 |
| --- | --- |
| `requirements.txt` | API/公开来源、XGBoost CPU 预测 |
| `requirements-experiments.txt` | scikit-learn、优化器、fpstreams 对照 |
| `requirements-scouting.txt` | 研究模型、追踪解析和绘图 |
| `requirements-tree-models.txt` | 可选 LightGBM、GOSS、CatBoost |
| `requirements-deep-learning.txt` | PyTorch 表示学习 |
| `requirements-sources.txt` | 可选 soccerdata 包装 |
| `requirements-vaep.txt` | socceraction 1.5.3 的独立环境，避免 NumPy 版本冲突 |

`python -m app.research` 是数据与研究入口，帮助、采集和公开导出不加载预测训练栈。`python -m app.pipeline` 保留原有生产训练、预测、导出和兼容研究命令。具体参数、数据前置条件和隔离环境见[复现命令](REPRODUCING.md)。

重新生成研究站点数据时使用以下命令。它只读取现有结果，不采集数据或重训模型。

```bash
python -m app.research export-research
```

## 系统结构

| 层 | 位置 | 职责与约束 |
| --- | --- | --- |
| 共享基础 | `app/config.py`、`app/file_io.py` | 配置和原子写入，不依赖应用命令或页面 |
| 数据 | `app/data_service/` | 来源访问、比赛归一化、追踪格式、快照和仓库，不导入 ML、分析或 Web 层 |
| 科研 | `app/ml/`、`app/analytics/` | 特征、模型、时序评价、战术和绘图，保留各自输入与可解释范围 |
| 应用 | `app/cli/` | 采集、球探/战术、模型研究与调度，重型依赖按命令延迟加载 |
| 发布 | `app/web/` | 按领域校验公开投影，再原子写入；不递归公开实验目录 |
| 前端 | `docs/` | 静态页面、共享交互和图表，只读取公开数据 |
| 验证 | `tests/`、`scripts/` | 原生测试、浏览器、依赖边界和公开清单检查 |

旧的比赛对象、异常、追踪读取及研究命令 import 保留兼容。移动代码不会改变坐标、时钟、概率类别顺序、时间留出规则或已有结果文件。新的实现位置见[实现报告](IMPLEMENTATION_REPORT.md)。

## 数据与产物

| 内容 | 位置 | 保留规则 |
| --- | --- | --- |
| 来源原件与下载凭据 | `.tmp/open-data/` 及明确的来源快照目录 | 属于输入，保留原字节、版本和日期 |
| 本地仓库 | `data/` | 与模型、来源原件分别管理，普通检查禁止打开主数据库 |
| 私有实验 | `.tmp/<run>/` 等已有运行目录 | 保留权重、逐场输出和复核证据，不能按 `.tmp` 名称一概清空 |
| 可公开结果 | `results/` | 聚合报告、来源身份和原创图，数值按各自运行解释 |
| 部署副本 | `docs/data/` | 明确白名单导出的 JSON 与图表，重复于结果目录属于必要部署副本 |
| 可再生缓存 | `__pycache__/` 等 | 验证与使用状态核对后可删除 |

原始数据、完整身份关联、个体预测、模型二进制和浏览器证据不进入公开清单。历史路径属于来源凭据，本轮没有为目录统一而移动原件或实验记录。详细覆盖和授权边界见[数据来源](DATA_SOURCES.md)。

## 检查与维护

先安装相应依赖。Node.js 和 Bash 用于 JavaScript 与 Shell 检查。

```bash
python scripts/check.py
python scripts/check.py tests.test_research_export tests.test_product_delivery
python scripts/check.py --static-only
```

默认入口运行全部 unittest，再检查全部前端脚本、页面资源、文档链接和公开文件身份。测试只使用受控样本及隔离文件，拒绝访问主 `data/football.sqlite3` 和 WAL/SHM。可选依赖按实际执行范围跳过，缺少优化器不会隐藏基础校准、比分或数值诊断。零测试收集和检查未完成会失败。

真实界面检查单独启用，需要可用的 Playwright 与 Chrome/Chromium。可用 `PLAYWRIGHT_MODULE` 和 `PLAYWRIGHT_CHROMIUM_EXECUTABLE` 指定本机路径，默认也识别现有本地运行时。脚本不自动安装组件，只启动并关闭本轮隔离浏览器和 localhost 服务。`FRONTEND_CASE` 可按名称筛选场景；未知名称或空选择会失败。

```bash
python scripts/check.py --static-only --browser
FRONTEND_CASE=entity-search-and-selection-restoration node tests/frontend_browser.cjs
```

新增或修改公开文件后，显式刷新并复核清单。刷新只发现源码与文档，不自动把新实验数据加入公开范围。

```bash
python scripts/public_manifest.py --refresh
python scripts/public_manifest.py
```

`checks.yml` 检查最小入口与研究环境。每日更新和 Pages 工作流保留独立职责。本地检查通过不表示远端 CI、模型发布或网站部署已经完成。

## 研究状态

保留的 49 篇论文及相关资料已完成约定阅读，无法补齐的三项按用户要求排除。适用实现、扩展候选和实际访问限制分别记在[实现报告](IMPLEMENTATION_REPORT.md)与[参考目录](results/resource-reviews.json)。闭源产品只借鉴公开资料，不声称取得订阅数据或复现内部模型。

[10 月 19 日离线结果](results/paper-followup-20261019)来自同一英超 2025/26 快照，包含比分、评级、校准、可选树搜索和固定输出解释。它们不替代既有五联赛结果、生产模型或当前赛季预测。历史实验细节与结论统一在[结果说明](results/README.md)，本轮维护和验证状态见实现报告。
