# 数据来源与采集状态

历史赛事和追踪样本沿用 2026 年 10 月 11 日的核对记录，2026/27 赛季数据在 10 月 14 日补采，ESPN 和 CSV 来源在 10 月 15 日增量复核。较早的采集范围按各自记录保留。数据保存在 `data/football.sqlite3`，包括原始文件、来源记录、解析后的表和分析结果。仓库中的数据集标签包含原始来源、镜像、访问检查和衍生结果，标签数量不等于独立数据来源数量。

10 月 14 日的更新取得欧冠 36 场及八个联赛 559 场完赛的 ESPN 摘要，另保留一场中止比赛。五大联赛 Understat 补齐了 297 场比赛的统计、8,440 条射门和 9,382 条球员单场记录。22 个 CSV 联赛当时共有 1,583 场本季赛果。国家队近期更新单独保存，未与俱乐部统计合并。OpenFootball 新抓快照的赛果仍只到 9 月 20 日，StatsBomb 和 SkillCorner 上游没有新增本季开放样本。完整数量、访问失败和验证范围见[本季更新清单](results/season-refresh.json)。

10 月 15 日重新下载了 66 个响应，复核 ESPN 的 10 月 13–15 日比赛和 22 个 CSV 联赛。CSV 新增 12 场英冠、1 场苏冠赛果，总数增至 1,596 场；欧冠修订了一处球队名称，未新增比赛或修改比分。30 场近期完赛的摘要均通过球队身份、主客场、结束状态和比分核对。新增内容已导入主库，原始响应另存快照。本次没有重采 Understat、OpenFootball 或国家队来源，也没有重训预测模型。

10 月 17 日的[采集重试改进](results/http-retry-policy.json)只调整请求失败后的处理，没有补采或入库。API、公开 CSV/JSON 和档案下载共用日期提示、等待上限与随机退避；档案路径不再对 403、404、501 重试。历史训练快照继续通过 `--refresh` 显式刷新，近期采集默认创建新快照，`--resume` 才复用已有采集目录。未启用自动过期缓存，也未用旧响应替代失败的最新请求，因此各处数据新鲜度仍按原采集日期阅读。

10 月 14 日增加的是[机构方法与参数的官方资料核对](results/METHODS.md)，以及使用已有 Wyscout 数据的[表示学习实验](results/REPRESENTATION_LEARNING.md)。本轮没有把方法页当作新赛事数据，也没有宣称此前受阻的平台已接通。研究页面统一展示 Understat 的赛季标记，例如 `2021` 显示为 `2020/2021`，来源原标记保留。

本轮以 [soccerdata](https://github.com/probberechts/soccerdata) 为采集入口，实际运行的是 PyPI 1.9.1，隔离环境保留了版本和源码哈希。它是采集和解析工具，数据仍来自 Understat、ESPN 等各个发布方。完整计数和访问状态见 [source-catalog.json](results/source-catalog.json)。

## 10 月 11 日的 soccerdata 采集结果

| 发布方 | 本轮取得的数据 | 状态与限制 |
| --- | --- | --- |
| [Understat](https://understat.com/) | 五大联赛 2014/15 至 2025/26，共 60 个联赛赛季组合。21,690 条赛程行、32,574 条球员联赛赛季汇总、21,589 条比赛统计 | 直接采集成功。赛程和统计的覆盖不同，不能将赛程行全部视为已结束比赛。转会球员的赛季汇总可能跨队，SDK 的球队标签只取第一支球队 |
| Understat 单场数据 | 英超 2024/25 的 9,878 条射门事件、11,567 条球员单场统计 | 直接采集成功，保留位置、用脚或身体部位、射门结果及供应商 xG 等返回字段。不是完整的传球与无球追踪数据 |
| [ESPN](https://www.espn.com/soccer/) | 英超 2024/25，380 场赛程、760 条球队比赛表、15,188 条阵容球员行 | 直接采集成功。阵容包含未出场替补。SDK 使用的旧 `boxscore.form` 已作兼容读取，按球队 ID 对齐主客场，原始 JSON 未改写 |
| [Football-Data.co.uk](https://www.football-data.co.uk/data.php) | 经 MatchHistory 读取器验证了英超 2024/25 的 380 场比赛 | 是已有 CSV 来源的另一条采集路径，不计为额外独立来源 |
| [SofaScore](https://www.sofascore.com/football/player/compare) | 未取得可入库的赛事或球员样本 | 普通匿名请求和 soccerdata 客户端均返回 403。1.9.1 的 SofaScore 类只实现联赛、赛季、积分榜和赛程，没有热图、射门图或球员单场统计方法 |
| [FBref](https://fbref.com/) | 直接浏览器读取未取得统计表 | 页面要求人工验证，测试停止。已有社区保存的历史 FBref 快照，见下文，不能据此说当前原站已接通 |
| [SoFIFA](https://sofifa.com/) | 直接浏览器读取未取得新数据 | 页面验证未通过。已有来自 Kaggle 和 GitHub 的游戏评分镜像。SoFIFA 游戏评分与 SofaScore 真实赛事平台分别记录 |
| [WhoScored](https://www.whoscored.com/) | 本轮未取得直接数据 | 较早的 SDK 联赛读取缺少 `allRegions`。2026-10-15 又在隔离浏览器访问已知单场页面，收到 Cloudflare 阻拦页，未取得 `matchCentreData`。社区历史文件另作存档 |
| [ClubElo](http://clubelo.com/) | 本轮未取得直接评分样本 | 原生地址返回 502。已有社区保存的 ClubElo 历史 CSV，不能当作此次直接采集成功 |

采集使用匿名会话和任务专用浏览器目录。验证码出现时停止，没有使用个人登录态或自动解验证码。原始缓存、表头结构、数据类型、请求状态和查询范围一起归档。

2026-10-15 对已有 Football-Data 赔率快照补做主平客分析。[官方列定义](https://www.football-data.co.uk/notes.txt)把普通列称为收盘前报价，带 `C` 的列才是收盘报价，不能把普通列当成开盘价。[官方质量提示](https://www.football-data.co.uk/data.php)说明 Pinnacle 从 2025-07-23 起的两个阶段都有滞后问题，市场均值也已排除该来源。本地参考分析因此排除这一日期起的 Pinnacle 比赛报价，并要求两个阶段的主、平、客三项全部有效，保留缺失和排除计数。结果见 [market-reference.json](results/market-reference.json)。本轮只分析已有本地快照，未新增该来源的自动采集或模型训练。

多赛季输入使用明确的起止范围。整数 `2021` 在库中可能被解释成 `2020/21`，因此采集器将起始年份转换为 `2021-2022`，并检查返回的赛季集合。修正后的 Understat 赛程覆盖十二个赛季，按来源比赛 ID 检查没有重复行。

后续分析又核实了三项口径问题。Understat 有 882 条球员赛季汇总包含多支球队，不能将整季数据全部归给 SDK 标签中的第一支球队。英超射门导出有 83 条点球情境被 SDK 映射为空，分析按原始射门 ID 恢复标签。ESPN 原始端点有 12 场比赛将双方控球率等统计都返回为零，这些占位值从控球分析中排除。Understat 球队汇总 xG 与逐射门 xG 之和也并非完全相同，射门质量分析统一从射门级数值计算。原始文件均未改写，结果见[新增数据分析](results/README.md#新增数据的分析发现)。

## 已有真实比赛与空间数据

主要战术研究聚焦男足五大联赛、欧冠、欧洲杯、世界杯和 2025 世俱杯。[核心赛事覆盖表](results/core-coverage.json)按来源、赛季和数据粒度记录实际范围。数据库里的澳超连续追踪和匿名 Metrica 样本用于方法验证，其他联赛保留为补充资料。

新增的 [FIFA 世俱杯报告索引](https://www.fifatrainingcentre.com/en/game/tournaments/fcwc/2025/post-match-summary-reports.php)公开提供 2025 年全部 63 场比赛的 PDF。[2026 世界杯小组赛](https://www.fifatrainingcentre.com/en/fifa-world-cup-2026/match-report-hub.php)与[淘汰赛报告](https://www.fifatrainingcentre.com/en/fifa-world-cup-2026/match-report-hub-knockout-stage.php)合计 104 场，也已匿名下载。报告、索引、来源地址、哈希及逐页文本存入同一本地数据库，数值另按球队、指标、单位和页码归整。原始 PDF 留在本地，公开目录只保留汇总和自行绘制的图。

这些是追踪系统衍生的报告级统计，包含控球阶段、防线高度、队形宽度与长度、压迫，以及球员分速度区间的跑动距离和最高速度。它们没有提供逐帧原始轨迹。两队的受控控球占比之外，FIFA 还保留双方都未控制球的争夺状态。阶段占比按原定义保留，不强制重新归一化。

体能表的次数标签在世俱杯报告中有变化：部分早期报告的 High Speed Runs 标为区间 4、Sprints 为区间 5，决赛则分别标为区间 3、区间 4 和 5。解析器按页保留标签，并以不同字段存储，不能直接拼成同一速度阈值的时间序列。距离区间和次数标签分别核对；这些比赛负荷指标也不足以测量恢复状态或伤病风险。

[Soccermatics-Skillcorner](https://github.com/soccermatics/Soccermatics-Skillcorner)提供英超追踪教学代码，README 描述的数据下载依赖课程 Dropbox 链接或 SkillCorner 账户，数据没有存放在仓库中。本轮仅作方法参考，没有将其写成已经采集的完整英超追踪赛季。

欧冠仍有明确覆盖缺口：StatsBomb 的历史档案每个已覆盖赛季仅有一场选择性比赛，并非全部都是决赛，也不是完整欧冠赛季。法甲 2015/16 的该档案有 377 场，德甲的部分赛季是单队样本；“存在赛季文件”不能视为整季齐全。

| 来源 | 已入库内容与范围 | 主要用途 |
| --- | --- | --- |
| [Football-Data.co.uk](https://www.football-data.co.uk/downloadm.php) | 1993/94 至 2026/27 的公开档案，22 个联赛或级别、711 个联赛赛季文件，242,538 场已结束比赛；部分文件含射门、角球、牌、裁判、观众和分阶段赔率 | 长期球队比较、赛季变化和比赛先验。最新赛季仍是部分赛季 |
| [OpenFootball](https://github.com/openfootball/football.json) | 多联赛的赛季 JSON，包括欧洲 CSV 之外的部分赛事 | 补充覆盖、比分和日期交叉核对。覆盖与 Football-Data 不同 |
| [StatsBomb / Hudl Open Data](https://github.com/hudl/open-data) | 3,961 场有日期元数据的比赛，24 个赛事编号、48 个赛季编号，日期从 1958 至 2025；事件、阵容、供应商 xG 和部分 360 可见空间 | 传球网络、动作区域、机会创造、压力行为、阵型变化和可见空间。公开档案具有选择性，不是连续完整的全球联赛样本 |
| [Wyscout 公开研究档案](https://figshare.com/collections/Soccer_match_event_dataset/4415000) | 五大联赛 2017/18、欧洲杯 2016、世界杯 2018；1,941 场比赛、3,251,294 个事件、3,603 名球员元数据、46,897 条 PlayeRank 球员比赛记录 | 球员习惯与位置比较、出场机会、推进、对抗和青年观察。当前球员评估实验采用五大联赛 2017/18 |
| [SkillCorner Open Data](https://github.com/SkillCorner/opendata) | 20 场澳超 2024/25 的 10Hz 追踪、动态与无球事件、比赛阶段，以及赛季汇总 | 轮转候选、转换负荷、团队宽度与深度、跑动类型。检测与外推分开，摄像机覆盖会影响分析 |
| [SkillCorner Body Pose](https://huggingface.co/datasets/SkillCorner/opendata-bodypose) | 两场完整比赛的 25FPS 姿态数据，29 个预测关节及不确定性信息 | 运动与姿态估计质量分析。是模型估计，不等于测量出的真实人体关节 |
| [Metrica Sports](https://github.com/metrica-sports/sample-data) | 三场匿名比赛的事件与连续追踪，含 CSV 与 FIFA EPTS 格式 | 轨迹、空间分析、速度估计和追踪方法验证。匿名球员不能直接与实名评分合并 |

Hudl 就是项目中使用的 StatsBomb 开放数据仓库。本次核对的提交与已归档版本一致，没有重复下载约 16GB 的事件档案。未引用到日期元数据的事件文件保留为原始文件，不进入时间切分分析。

## 本轮增加的社区存档

[jokecamp/FootballData](https://github.com/jokecamp/FootballData) 和 [eddwebster/football_analytics](https://github.com/eddwebster/football_analytics) 的数据文件已固定版本下载并校验。共登记 9,667 个文件，识别出 2,093 个逐字节相同的镜像，复用一个原始内容。共涉及 6,072 个不同的 CSV 内容，按表头组合存入 259 种表结构。表行数、文件数和独立比赛数是不同计数。

| 社区存档 | 实际内容 | 分类 |
| --- | --- | --- |
| jokecamp | MLS 历史、世界杯与国家队、英超等联赛资料、球员与名单 JSON、Football-Data 镜像，以及 FIFA12 游戏数据 | 混合历史存档。年份和原始发布方逐文件核对，不能把全部文件视为独立来源 |
| Edd Webster 的 FBref 文件 | 五大联赛 2017/18 至 2020/21 的场上球员和门将统计，以及合并、工程化版本 | 社区保存的历史快照。原始与衍生文件分开，重复内容不叠加 |
| Edd Webster 的 Understat 文件 | 多联赛历史射门与比赛元数据 | 历史镜像，与本轮直接采集的 Understat 分开标注 |
| Edd Webster 的其他目录 | ClubElo 历史、Opta 标签事件、Stats Perform 加拿大联赛样例、Stratabet 机会数据、Capology 薪资、Transfermarkt 转会与估值，以及其他表现资料 | 目录标签表示社区声明的上游，不能解释成这些机构已被直接接通。不同字段和许可需分别核对 |

CSV 的原始表头、缺失值和异常行被保留。XLS/XLSX 已按原始文件入库，未当作已经解析、可直接用于模型的 CSV。仓库公开可见不等于原始发布方授权再次公开分发，因此原始存档留在本地。程序未执行这些仓库中的采集脚本或 notebook。

jokecamp 存档中的 `espn/players0-50.json` 是空文件，已隔离并保留原件。原有三份隔离文件还包括 StatsBomb 360 和 Wyscout 裁判档案；新增报告的解析状态另列在 [FIFA 入库汇总](results/fifa-reports.json)，隔离内容不进入解析后的分析表。

## 游戏评分与数据库

| 来源或渠道 | 已取得的数据 | 注意事项 |
| --- | --- | --- |
| 本地 FM23/FM24 解码件 | 23.0.0、23.3.0、23.4.0、24.0.0、24.3.0 五份快照，2,341,262 条球员快照记录；同时保存人物、俱乐部、职业和雇佣历史及原始审计资料 | 快照记录不是不同球员人数。原生属性、CA、PA 与负潜力范围码各自保留 |
| Kaggle FM 数据 | FM17、FM19、FM20 和覆盖 FM20 至 FM23 的界面导出 | 导出格式和百分比角色评分不同于原生 CA。缺少有效日期的导出不能直接进入历史预测 |
| Kaggle FIFA/EA FC 数据 | FIFA15 至 FIFA23 的多次阵容更新、FC24，以及 FC25/FC26 导出；评分记录超过千万条 | 更新快照不等于千万个不同球员。评分是游戏估计，需要版本与日期约束 |
| GitHub 游戏资料 | Radzim 的 FC25/SoFIFA 镜像、07fahim 的 FIFA12 至 FC25 精选球员样本、FM24 角色衍生资料及 jokecamp FIFA12 文件 | 精选样本不是完整球员库，标准化角色特征不是原生 1至20 属性 |
| [EA 评分端点](https://www.ea.com/games/ea-sports-fc/ratings) | 保存了公开端点的评分快照 | 当前端点返回与网页有差异，版次和生效日期没有确认，不用于需要历史日期的验证 |
| [FMInside](https://fminside.net/players) | 来源状态和 FMUpdate 的说明参考 | 直接访问与下载返回 403，没有取得完整球员库或模组。网站 0至99 数值与原生 FM 属性分开；社区更新不当作原版快照 |

Kaggle 和 GitHub 是获取渠道。各数据集的下载地址见[游戏数据配置](app/data_service/ratings.py)，数据库同时记录作者、原始 URL、游戏版次、采集时间和哈希。游戏评分、真实比赛统计和分析模型输出分别保存。

## 六个指定仓库的使用结论

| 仓库 | 本次使用 | 与足球研究的关系 |
| --- | --- | --- |
| [probberechts/soccerdata](https://github.com/probberechts/soccerdata) | 隔离安装、实际采集、访问测试和字段结构保存 | 作为采集入口，复用缓存和解析器。不能把“存在某个来源类”当作访问已成功 |
| [hudl/open-data](https://github.com/hudl/open-data/tree/master/data) | 核对现有归档与版本 | 已有事件与 360 数据，属于同一个 StatsBomb 来源 |
| [jokecamp/FootballData](https://github.com/jokecamp/FootballData) | 校验并导入数据文件，读取解析代码作参考 | 补充历史赛事与名单，保留镜像关系 |
| [eddwebster/football_analytics](https://github.com/eddwebster/football_analytics) | 校验并导入存档，静态阅读工程化 notebook 和关键脚本 | 除数据外，提供空间控制和 EPV 的方法参考。`Metrica_PitchControl.py` 的拦截时间与控制面函数、`Metrica_EPV.py` 的价值计算值得后续用真实追踪验证 |
| [google-research/football](https://github.com/google-research/football) | 归档观察字段、API 和反击、配合场景代码，静态阅读 | 是已归档的强化学习模拟环境。可研究可控场景，但本轮没有安装引擎或生成仿真样本；其疲劳、位置和奖励变量不是真实球员数据 |
| [architv/soccer-cli](https://github.com/architv/soccer-cli) | 静态阅读请求与配置代码 | 是比分终端工具，主要依赖 football-data.org 的旧接口和 API token，不是新的免 key 原始数据库，本轮未运行 |

FIFA Training Centre 的赛事赛后报告已成为数值来源，其方法文章和赛前观点仍单独标记。UEFA 技术与体能报告、FM 官方说明另作方法与图表参考。友利 Mori 与司马尘的 B 站字幕仍未取得；标题、简介和访问诊断不计为已读取的战术分析或数值样本。

## 复现入口

采集依赖使用独立环境，保持当前预测与分析环境的版本。

```bash
python3 -m venv .tmp/sources-venv
.tmp/sources-venv/bin/python -m pip install -r requirements-sources.txt

.tmp/sources-venv/bin/python -m app.data_service.soccerdata_collect Understat \
  --output .tmp/open-data/soccerdata/Understat-history \
  --leagues 'ENG-Premier League,ESP-La Liga,FRA-Ligue 1,GER-Bundesliga,ITA-Serie A' \
  --seasons 2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025 \
  --request-budget 180 --deadline 900

.tmp/sources-venv/bin/python -m app.data_service.soccerdata_collect Understat \
  --output .tmp/open-data/soccerdata/Understat-details --seasons 2024 \
  --details --request-budget 500 --deadline 1200

.tmp/sources-venv/bin/python -m app.data_service.soccerdata_collect ESPN \
  --output .tmp/open-data/soccerdata/ESPN --seasons 2024 \
  --details --all-matches --request-budget 500 --deadline 1200

python -m app.data_service.soccerdata_ingest
python -m app.pipeline collect-open-data --datasets jokecamp,eddwebster
python -m app.data_service.repository_data
```

SoccerData 读取器保留原始网页或 JSON 缓存、来源内标识符、多层表头和访问报告。当前截止控制使用 Linux 的信号机制。失败的来源报告单独作为访问检查保存，不能计成成功的比赛或球员数据。

新增数据没有自动合并到已有预测模型或青年名单中。跨来源实体、日期、坐标单位、出场分母和 xG 定义仍需核对，历史镜像和工程化表也不能与原始表重复计数。

## 10 月 11 日增补

[Mart Jürisoo 国家队档案](https://github.com/martj42/international_results)固定到提交 `d5bd6de012c87e2f325de602ae77fe2f9ab61ef3`，包括赛果、进球者、点球决胜和队名变化资料。原始结果 49,945 行，两行比赛身份有冲突，规范表保留 49,943 条。源数据的比分含加时，不含点球决胜比分，中立场单独记录。队名使用整理后的继承者名称，因此不是所有历史政治实体的原样名称。

ESPN 近期端点按实际比赛日读取，范围为 9 月 20 日至 10 月 11 日，取得 513 条赛事记录和 478 份已结束比赛摘要。俱乐部、国家队、性别和 U21 范围分开。Football-Data 当前赛季也已刷新核对，原有历史文件保留。Hudl 最新提交仍与已有版本相同，没有重复下载事件库。近期国家队比赛有赛果和部分统计，不等于已经取得完整无球追踪。

新增[战术资料索引](results/tactical-evidence.json)区分文章全文、自动字幕、作者提供的播客转录、节目说明和访问失败。Jesse Marsch 视频英文字幕已取得，B 站两位作者的详情接口仍返回 412，The Athletic 所选音频端点返回 403。来源自述与数据结论分别存储，教练任职也不从过期 roster 快照推断。

新增[转移进攻侧研究](results/switching-play.json)复用 StatsBomb 的 2024 欧洲杯 51 场事件文件和对应赛事元数据。它是已有来源上的新分析，不增加独立来源数量。跨场传球只是球的移动代理，教练与对手按比赛当天关系关联，不能代替无球追踪或解释为因果效果。

安全修复后，通用入库复用 SoccerData 的公开文件选择规则，拒绝目录与硬链接别名，浏览器运行态不入库。Wyscout 的来源文件名必须是平面名称，下载哈希合格后才落为完整文件。修复前检查真实数据库的资产目录，没有发现 Cookie 或运行日志；临时哨兵测试证明的入口问题已修复。模型包仍只供可信的本地训练流程使用。

转移情境新增 SkillCorner 官方动态事件规范和压迫指标说明，战术参考索引现有 20 个来源、26 条卡片。澳超情境复用已有 20 场追踪与动态事件，逐次保留成功/失败、可见性、球权变化和教练缺失，没有新增跨来源比赛合并。欧洲杯 360 帧用于静态站位，澳超连续检测轨迹用于移动，两者按各自坐标与范围解释。

## 10 月 19 日历史位置数据增补

为 2014/15 至 2024/25 五大联赛位置变化研究补入了 48 个公开存档导入条目，共 436,772 条解析行。它们包含 FBref、WhoScored 的不同统计表及部分赛季诊断快照，不是独立球员数。完整 2022/23 与 2023/24 存档补回了此前缺少的统计，2014/15 至 2016/17 恢复了防守、成功过人和传球字段。早期推进传球、推进带球、禁区触球和射门创造动作仍缺失，保留为空。原始来源、固定提交、哈希与文件内容保存在数据库的 `position_supplement_sources`、`assets` 和 `raw_chunks` 中。

Understat 为全部 8,901 名研究球员返回了历史接口，按既有赛程 ID 筛出 575,896 条球员比赛记录，保存在 `position_history_matches`。历史标签来自逐场位置，个人资料的当前 favorite_position 不用作历史标签。赛程范围有 19,938 个来源比赛 ID，这个数包含赛程记录，不能都称作完赛。

一场比赛有 30 条同队重复名单，重新核对比赛端点后只保留一次出场。冲突的事件字段保留 NULL，原始变体和比赛核对资产保存在 `position_history_issues`。赛季汇总来源也受重复名单影响，分析输出修复了分钟和场次，原始汇总资产没有改写。来源 ID 对照及明确别名修复了 399 条球员赛季关联，仍有部分跨来源姓名匹配需要进一步核实。

研究主表保留 8,901 名球员、29,217 条球员赛季记录，没有因补充字段缺失而删除整季数据。新增数据尚未接入现有应用模型，也没有重训预测模型。数据集标识、表名、计数、缺口和本地验证见 [历史位置数据清单](results/position-transition-20261019.json)。


## 存储与生命周期

采集原件、规范化比赛、仓库、私有运行和公开摘要是不同阶段。`.tmp/open-data` 和带日期的来源目录目前沿用历史路径，其内容含后续复现所需的输入，不能按临时目录统一清空。比赛归一化由 `app/data_service/match_records.py` 负责，追踪格式由 `app/data_service/tracking_frames.py` 负责，分析层只消费其时钟与坐标契约。

`data/football.sqlite3`、WAL/SHM 和已归档来源文件不属于普通维护清理目标。自动检查通过拒绝访问钩子防止误用主库，数据库相关测试使用临时隔离库。已有输入快照、私有逐场输出、模型与运行编号继续保留，不能把删除后重新下载视为同一输入版本。

`results/` 保存选定的公开结果，`docs/data/research/` 保存经过字段白名单和一致性校验的部署副本。即使两份内容逐字节相同也应保留两者的职责。公开清单的固定核对入口为 `python scripts/public_manifest.py`，刷新必须显式传入 `--refresh`。该操作不会扫描来源档案或把新私有实验自动公开。
