# 数据采集与研究复现

新版入口、目录与表示学习实验见[项目 README](README.md)。下列命令保留已有数据采集和评估流程，比赛数据数量按各处标注的快照日期阅读。10 月 14 日的重构与方法研究没有把 10 月 11 日的比赛快照改写成最新赛果。

研究页四项摘要分别加载和重试。验证时可在隔离浏览器中让其中一个摘要返回 HTTP 503，再恢复请求并用键盘重试，检查其他区域的内容、筛选和请求次数保持不变。损坏 JSON、缺少必要字段和绘图中途异常也应显示区域内错误。本轮证据保留在本地 `.tmp/affordable-football-20261015`；模拟故障只作用于测试页面，没有修改公开数据文件。

本轮保留49篇论文的全文、所列附录和八份独立补充材料已读完；编号06、L18、L22按用户要求排除，实际阅读范围保留。10月19日收到“开始实现”后，已完成十项缺口及既有适用修正，再集中测试。热图网格/平滑/单位、固定分量来源和TreeSHAP已落实；没有将三传球计数称为随机网络Z分数，也未重新执行本地70%解释方差选择。具体证据和限制见[实现报告](IMPLEMENTATION_REPORT.md)。

Soccer Analytics combines key-free football results, event and tracking data with dated game ratings in a local SQLite warehouse. It evaluates players and teams, studies chance creation and opponent matchups, and links observed running to positional exchanges and transitions. Match prediction and the Touchline dashboard remain available. XGBoost is the default match predictor; research experiments compare several model families with chronological holdouts.

See [research results](results/README.md) for the measured improvements, regressions, figures and coverage limits. The public repository contains code and aggregate results. Raw provider data, local FM conversions and fitted model binaries remain local.

The primary tactical scope is the men's Big Five leagues, Champions League, European Championship, World Cup and 2025 Club World Cup. The [competition coverage](results/core-coverage.json) records source, season and data grain; A-League and anonymous Metrica tracking samples support method validation. Selected historical Champions League games do not provide complete modern seasons.

[传球输入时点复核](results/pass-input-timing.json)记录了共享序列编码器的输入、标签和时钟约束。`pass_sequences` 要求同场、同半场内事件时钟不倒退，正常加载器按 `period, seconds, row_number` 排序。同刻记录保留来源顺序，不能视作实时可用顺序的证明。本轮只读审计脚本和逐场记录保留在本地 `.tmp/pass-feature-timing-20261015`，没有训练或发布模型。

[传球样本筛选复核](results/pass-cohort-eligibility.json)对应原表示学习运行 `f94ded4a65f37938`。`pass_sequences(events, audit=counts)` 仍返回原来的三个数组，成功编码后向字典写入本次计数，空输入也写入零。`load_dataset` 的每个分区新增 `label_eligibility`，统计抽样前的事件与标签条件；原有 `completion_rate` 仍是抽样后的完成率。排除原因按起点有效性、终点数量、结果标签、终点有效性的顺序互斥计数。起点检查包括坐标、时钟和事件类型；倒退时钟直接报错。有效起点会留在后续上下文中，即使该事件自身没有可用标签或终点。

该审计只涵盖原实验的 1,080 场常规时长比赛、上下半场、名称严格为 `Pass` 的事件；不是五大联赛全部场次，也不包括其他名称的任意球事件。四个分区共 924,913 次 `Pass`，各项排除数为零。逐场输入和标签与本轮改动前一致，新增计数以外的完整加载记录也与原运行一致，包括比赛和采样行归属。模型未重新训练或评分。只读脚本、对照快照及私有逐行记录保留在 `.tmp/pass-cohort-eligibility-20261015`；本机可用 `PYTHONPATH=. .venv/bin/python .tmp/pass-cohort-eligibility-20261015/audit.py` 复核。该脚本依赖本机基准快照和原私有实验记录，不是公开仓库的独立复现入口。

[五联赛固定参数对照](results/representation-leagues.json)沿用现有编码、抽样、训练、评分及整场配对重采样。只训练目标联赛模型，不做来源预训练，也不按新增联赛的测试结果调参。完整私有记录和预测在 `.tmp/representation-leagues-20261015/run`，汇总不含逐行预测或比赛ID，图已加入公开图表库。输出目录必须为空，避免覆盖旧实验。

```bash
.venv/bin/python -m app.analytics.representation.leagues --database data/football.sqlite3 --output .tmp/representation-leagues-repeat --train-cap 8000 --epochs 15 --seeds 11 29 47
```

本轮运行 `76efac828d6b93d5` 的训练源码快照保留在本地 `.tmp/representation-leagues-20261015/execution-code`。训练完成后只修正了图的脚注布局和运行编号副标题；汇总保留原执行代码指纹，`rendering` 另记录最终绘图代码指纹，去掉绘图函数后的源码AST一致。重画图不重新训练。原英格兰12方法实验、模型权重及分数均保留。

[XGBoost早停复核](results/early-stopping-kickoffs.json)使用已归档的三季英超数据，比较内部按行切分与保留完整开球时间组的差异。`XGBoostClassifier.fit(..., kickoffs=...)` 接受与行对应、按时间排序且带时区的 ISO 字符串序列；等价时区先转为 UTC。省略参数或传入空序列保留按行切分的行为。需要早停却无法形成独立时间组时会报错。普通训练、回测、参数搜索与赛季矩阵通过共享拟合入口传递时间，其他估计器仍使用原来的双参数 `fit`；重写 XGBoost `fit` 的子类需要接收并转发 `kickoffs`。

该复核保留默认质量检查。另做的两处拟合诊断不计算外层留出集的预测指标，也不保存模型。早停仍计算内部验证损失，全历史诊断的验证后缀和完整重拟合都包含原外层留出比赛；这些诊断不作为模型发布依据或预测效果证据。完整脚本和记录留在本地 `.tmp/early-stop-kickoffs-20261015`。此前报告对应此前代码，不按新协议改写。开球时间先后也不能替代终场时刻或数据到达时间的检查。

The [Chelsea–PSG final case](results/README.md#世俱杯决赛切尔西-30-psg) compares the two teams' six earlier tournament games with their 2025 final. It combines official report metrics with dated coach statements and specific attacking sequences. Coaching intent, observed behavior and the effect of a 3–0 half-time lead remain separate. FIFA's 2025 Club World Cup and 2026 World Cup reports add shape, pressure, phase and physical data; they do not supply raw synchronized player trajectories.

```bash
# Anonymous public reports; requires Poppler's pdftotext for ingestion.
python -m app.data_service.fifa_reports collect --series fcwc2025
python -m app.data_service.fifa_reports collect --series fwc2026
python -m app.data_service.fifa_reports ingest --series fcwc2025
python -m app.data_service.fifa_reports ingest --series fwc2026
python -m app.analytics.elite_tactics
```

已有仓库和核心赛事视图时，可只读重建覆盖表。这个命令只生成覆盖 JSON，不重新采集、不改写数据库，也不重做战术案例和图。输出日期是本次覆盖核对时间，不能当作所有来源的重新采集时间。

```bash
python -m app.analytics.elite_tactics --coverage-only --database data/football.sqlite3 --output .tmp/coverage-check
```

ESPN 摘要数以已完赛来源记录为分母。匹配要求摘要完整归档，且比赛编号、主客队身份、终场状态和比分与最新赛果一致。`missing_summary_documents` 包括没有关联、没有文档、归档不完整或文档类型不符，`invalid_summary_documents` 表示格式错误或与赛果不符。三个摘要计数合计等于该行来源记录数，不保证摘要内所有统计字段齐全。

The [source catalogue](DATA_SOURCES.md) records collected data, coverage and access failures, and distinguishes original publishers, community mirrors, collection tools and simulation environments. The optional SoccerData collector adds multi-season Understat data and ESPN match rosters in a separate environment.

The [new-source findings](results/README.md#新增数据的分析发现) compare player repeatability, possession and chance quality, and changes in team attack and defence. Run `python -m app.analytics.new_source_findings` after importing the SoccerData frames. The analysis restores penalty labels from raw shot IDs, excludes invalid possession placeholders and keeps multi-club player totals separate from team aggregates.

The [October refresh and coach context](results/README.md#10-月-11-日比赛更新与教练背景) adds national results, dated coach observations and tactical reading/caption cards. Recent collection defaults to the last three weeks; research imports and coach analysis remain separate from the production `all` command.

## Project layout

```text
app/
  config.py             # Environment settings and competition selection
  pipeline.py           # Prediction, collection and offline research commands
  data_service/fetch/   # API and key-free CSV/JSON sources with rate limiting
  data_service/         # Source archives, game ratings and SQLite warehouse
  analysis.py           # Team statistics with Python and optional fpstreams
  analytics/            # Player/team assessments, tactics, tracking and figures
  ml/                   # Features, training, evaluation and model bundles
  web/                  # Dashboard export and release checks
docs/                   # Static dashboard and exported JSON
results/                # Aggregate research results and reproducible figures
models/                 # Local model bundles and previous versions
tests/                  # Python tests and JavaScript runtime checks
```

## Football observation and scouting

The warehouse also supports questions about how teams play, where players receive
and move the ball, which habits recur across matches, and which young players need
further observation. Prediction models are one research tool in this workflow.

```bash
# After building the warehouse and running assess/tactics:
python -m app.research scout-insights --backend compare
python -m app.research scout-insights --player-as-of 2018-02-28 --output-dir .tmp/scouting-february
python -m app.research scout-figures --match-id 3942226 --network-window 10 40 --output-dir results/football
```

`scout-insights` builds ball-action location grids, attempted pass direction,
crossing tendencies, tagged shot-foot usage and recurring match patterns. Cards
retain event IDs and clocks for review. A reported preferred foot and observed
shot-foot distribution are different fields. Missing foot tags stay unknown.

The youth longlist uses exact calendar ages16-21 at a historical cutoff and starts
at90minutes, keeping limited exposure visible. Manual review flags require450minutes,
five matches and a strong observed role/league dimension. Age/role comparisons stay
within each league and require adequate peer counts. Four within-season windows
describe opportunity and observed changes. Real future potential and biological
maturation remain unknown. Dated FIFA overall/potential values are contextual
references and do not select the longlist. The current sample cannot validate a
long-term growth forecast.
Roster roles and reported preferred foot have no dated historical snapshots, so
early-cutoff cards remain retrospective descriptions.

Opponent preparation compares the same team against one opponent and its other
opponents within a competition-season, aligning home/away fixtures. It retains
formation choices, direction, possession proxies, chance output and evidence
matches. Remaining opponent strength and score differences limit interpretation.

Results are under `.tmp/scouting-intelligence` and in `player_habit_cards`,
`youth_observation_watchlist` and `opponent_preparation_cards`. Generator-based
cohort summaries have Python/fpstreams parity checks with explicit action weights
and missing-value denominators. These checks establish aggregation correctness,
without a training or performance acceleration claim.

See [football observations and figures](results/README.md) and
[tactical reading references](results/tactical-reading.json). The latter records
selected Bilibili creators, which metadata was available, and which
commentary remains unread. FM supplies familiar chart forms; the pitch graphics
use verified real event/tracking data. Passing-network arrows preserve direction,
and the published example selects first-half minutes10-40 to limit substitution
mixing. Continuous tracking occupancy, event-time visible positions and pass
origins have separate interpretations.

## Run with Docker

Use Docker Compose v2.24 or newer for the optional `.env` file syntax in `docker-compose.yml`. Without an API key the pipeline uses public CSV data. To use football-data.org, set `FOOTBALL_DATA_API_KEY` in a local `.env` file. Then run:

```bash
bash run.sh --competitions PL --days 3
```

The script builds the image, trains the selected models and exports the dashboard. `rebuild.sh` runs the same flow. Models persist in `models/`, dashboard data in `docs/data/`, and logs go to the terminal. Each command runs in a disposable container:

```bash
docker compose run --rm app train --competitions PL
docker compose run --rm app predict --competitions PL --days 3
docker compose run --rm app export-site --competitions PL --days 3
docker compose run --rm app all --competitions PL --days 3
```

## Run locally

Use Python 3.12 or newer and install the dependencies in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m app.pipeline all --source football-csv --competitions PL --days 3
```

Local Python reads exported environment variables; Docker Compose loads `.env`. The pipeline uses in-process rate limiting and local model files. `predict` writes a JSON result to standard output, including any competition errors; logs go to standard error.

Choose competitions supported by your selected source. The default selection is all codes in `app/config.py`. For production prediction commands, the public sources support eight of those configured codes, listed below. Historical research accepts further divisions described later. Optional settings:

| Variable | Default |
| --- | --- |
| `SOCCER_ANALYTICS_DATA_SOURCE` | `auto`: football-data.org with a key, public CSV otherwise |
| `SOCCER_ANALYTICS_COMPETITIONS` | All configured competitions |
| `SOCCER_ANALYTICS_TRAINING_SEASONS` | Four European season start years, including the current season |
| `SOCCER_ANALYTICS_PREDICTION_DAYS` | `3` |
| `SOCCER_ANALYTICS_SITE_EXPORT_DAYS` | `1` |
| `SOCCER_ANALYTICS_REQUEST_TIMEOUT` | `15` seconds |
| `SOCCER_ANALYTICS_REQUEST_RETRIES` | `3` |
| `SOCCER_ANALYTICS_MODEL_DIR` | `models` |
| `SOCCER_ANALYTICS_SITE_DATA_DIR` | `docs/data` |

Use comma-separated values for competitions and season years. `--competitions` selects from enabled competitions; `--seasons` overrides training years. For calendar-year leagues and international tournaments, select the applicable seasons explicitly. Training uses only finished matches. `--days 1` covers the remainder of today in UTC; `--days 3` also includes the next two UTC dates. Predictions include future `SCHEDULED` and `TIMED` fixtures in that range.

## Key-free data sources

| `--source` | Data | Authentication |
| --- | --- | --- |
| `football-csv` | [Football-Data CSV](https://www.football-data.co.uk/data.php): seasonal results and current fixtures | None |
| `openfootball` | [OpenFootball JSON](https://github.com/openfootball/football.json): seasonal results and available fixtures | None |
| `football-data` | football-data.org v4 API | `FOOTBALL_DATA_API_KEY` |

The configured production codes shared by both public adapters are `PL,ELC,BL1,PD,SA,FL1,PPL,DED`. They validate input and attach source URLs, time precision and time zone assumptions. CSV kickoff times use the UK clock; OpenFootball times use the league's local clock. Missing times are grouped at UTC midnight for historical features. Date-only future fixtures are omitted because their kickoff cannot be verified.

Providers have different team names, coverage and IDs. Training uses one provider, records it in the bundle, and prediction rejects a different provider. Retrain when switching sources. OpenFootball may omit matches or include playoff games; it is useful as an alternative dataset, not an exact substitute for CSV. Public fixture files may cover only the next round or contain no fixtures in your requested window. An empty result does not establish that no matches exist.

```bash
python -m app.pipeline train --source openfootball --competitions PL --seasons 2022,2023,2024
python -m app.pipeline predict --source openfootball --competitions PL --days 3
```

## Analysis and model experiments

### HTTP retries and snapshot freshness

The API, public CSV/JSON clients and archive downloader share a retry policy for
connection failures and HTTP 429, 500, 502, 503 and 504. Other HTTP errors stop
immediately. Valid `Retry-After` seconds or HTTP dates take precedence over local
backoff and are never shortened by jitter. Without a valid hint, each wait is
sampled uniformly from zero to the capped exponential delay. The default cap is
60 seconds; clients start at 0.25 seconds and archives at 1 second.

`FootballDataClient` and public clients accept `max_retry_delay`, `retry_jitter`,
`wall_clock` and `random_value`. The latter two make clock and jitter checks
repeatable. Client retry counts remain `max_retries + 1` attempts; archives keep
four attempts, including each call to `before_request`. Failed responses close
before waiting; a successfully returned public response or archive stream belongs
to its caller.

When attempts remain, a server hint above the cap stops the call without another
request: clients raise `DataSourceError`, and archives raise
`RetryAfterLimitError`, a `requests.RequestException` subclass. Both expose
`retry_after_seconds`. When a retryable failure exhausts the attempt budget,
clients retain `DataSourceError` and archives retain the original requests
exception, with any parsed hint attached and no additional wait. These values
let a caller decide when to try a new call;
there is no background scheduler.

The [audit receipt](results/http-retry-policy.json) records the local checks and
the selected [Tenacity wait source](https://github.com/jd/tenacity/blob/32ac68f0941319c0d2f4fb927f96e4ee29bb46c6/tenacity/wait.py).
HTTP-date handling follows [RFC 9110](https://www.rfc-editor.org/rfc/rfc9110.html#section-10.2.3),
including its three date formats and rolling two-digit year rule. Fractional
nonnegative seconds remain accepted for compatibility with the previous client.
Neither Tenacity nor requests-cache is installed. The latter's
[expiry and stale-response options](https://requests-cache.readthedocs.io/en/stable/user_guide/expiration.html)
were reviewed without enabling HTTP caching. Snapshots stay pinned unless `--refresh` or an explicit collection lifetime is
supplied; recent collection creates a new snapshot unless `--resume` is supplied. A retry change does not refresh archived football data.

### Explicit snapshot lifetimes

```bash
python -m app.research collect-history --sources football-csv --competitions PL,FL1,BL1,SA,PD --seasons 2026 --active-max-age-hours 6 --cache-dir .tmp/current-history --output .tmp/current-history-receipt.json
```

Both `collect-history` and the football-csv branch of `collect-open-data` accept
`--max-age-hours` and `--active-max-age-hours`. Neither has a default lifetime:
existing verified snapshots remain pinned. The active lifetime overrides the
global lifetime for the current and future European starting years, using UTC
July rollover. Zero expires immediately; age equal to the lifetime has expired.
`--refresh` forces a download regardless of the lifetime. Invalid, missing, or
future fetch timestamps cannot satisfy a finite lifetime.

Receipts record the cache decision, prior age, applicable lifetime and check
time. That decision describes the input cache; `fetched_at` describes the output
snapshot. Pinned is distinct from fresh. A failed refresh excludes the failed
collection and retains its previous snapshot bytes, without a stale-success
fallback. Missing snapshots are collected normally.

Bulk ZIPs have checksum- and URL-bound fetch receipts. A downloaded ZIP passes
CRC validation before atomic replacement. ZIP refresh also rebuilds normalized
JSON; receipt-backed reuse requires a matching archive hash. A valid ZIP may be
stored while a league member fails normalization, so this is not an all-league
transaction. Failed members remain errors, including on the next cache hit.
When deriving JSON from a pinned legacy ZIP without a fetch receipt, retrieval
time is unknown (`fetched_at: null`); `normalized_at` records local processing.
With a finite lifetime, that ZIP must be downloaded before reuse.

The [audit](results/snapshot-freshness.json) includes 11 actual loopback HTTP cases
with a controlled clock, an independently checked five-league 2026/27 collection,
and a second CLI run with zero network requests. It preserves 15 historical
source receipts. The current CSVs contain 297 completed matches, with latest
kickoffs on October 11–12; a new download does not prove current-day completeness.
The new data remains in an isolated snapshot directory; no warehouse or model
is replaced.

This native policy adapts selected [requests-cache expiration documentation](https://requests-cache.readthedocs.io/en/stable/user_guide/expiration.html).
It does not execute that library or implement HTTP conditional requests, server
Cache-Control semantics, stale fallback or asynchronous refresh. Bulk collection
still downloads the index page to discover archives.

### Pinned OpenFootball data

```python
from app.data_service.fetch.public import OpenFootballClient

client = OpenFootballClient(revision="e6744429ee395bc86f247348c6184bb08d4eb361")
fixtures = client.fetch_matches("PL", season="2026")
```

An optional full 40-character commit SHA pins the JSON URL. Without it, the
client keeps using `master`. Match and team IDs remain scoped to the provider
and league, independent of the commit. The adapter preserves round labels,
optional half-time scores and raw extra score codes. Invalid half-time values
retain the fixture, with nullable values and a quality flag. Half-time totals
above full-time totals are retained and flagged. Additional score codes do not
replace `score.ft`. Responses close after success or failure.

Scheduled queries enumerate every season between their endpoints using the
existing July starting-year convention. Date-only fixtures remain `SCHEDULED`;
the timed scheduling interface continues to require an actual kickoff time.
Missing scores on past dates do not establish that a match is still upcoming.

The [audit](results/openfootball-review.json) reads all eight 2026/27 league
files at one commit: 2,916 fixtures, 470 recorded results and 441 half-time
pairs. It checks 17,496 core fields, 882 half-time values, 470 unchanged
pre-match feature rows and 24 lossless exports. Each raw file also matches
its Git blob hash in the pinned directory. Five explicit name aliases allow
all 250 results from the five major leagues to match the earlier Football-Data
CSV snapshot, with no score conflict or provider ID merge. The Championship,
Eredivisie and Primeira Liga have no cross-provider comparison in this audit.
Unknown club changes remain unmatched.

The pinned JSON catalogue contains eight league files for 2026/27 and no
Champions League file. The separately pinned Champions League source has no
2026/27 directory. Recorded results in all eight league files end on
September 20 despite retrieval on October 18. The upstream
[update notes](https://github.com/openfootball/football.json#frequently-asked-questions--answers)
distinguish daily JSON generation from updates to its source data. This capture
does not refresh production models or the website.

### Provider-labelled match xG

```bash
python -m app.data_service.database --database .tmp/xg-provenance-20261017/warehouse-final.sqlite3 --xg-coverage
```

The CSV normalizer retains `HxG` and `AxG` as nullable `home_xg` and `away_xg`
statistics when either source column exists. Metadata distinguishes an absent
column, a missing cell, an invalid value and a valid value. Zero is valid;
finite nonnegative totals can exceed one. Invalid optional statistics retain
the fixture, and historical files without either column keep their old shape.

The command reads complete football-csv source assets in one read-only SQLite
snapshot. It reports coverage and paired summaries per asset, keeping revisions
separate. Summaries require consistent field metadata, two valid values and a
FINISHED fixture. It does not infer provenance for legacy values. A missing
database fails without creating a new file. Existing archived snapshots remain
pinned; updating the parser does not rewrite them automatically.

The [audit](results/match-xg-provenance.json) reparses the previously downloaded
297 fixtures offline, checks all 594 xG values against source cells through an
independent SQL projection, and verifies 15 lossless asset exports. Derived
snapshots retain the observation time of their exact raw CSV bytes, with a
separate local normalization time. Original snapshot bytes and all 297 rows of
the existing 30 pre-match features remain unchanged.

[Cavus and Biecek, Section 2](https://arxiv.org/html/2206.07212v2) distinguish
shot probabilities from aggregates. Here the provider's xG model, penalty
convention and exact availability time remain unverified. The stored usage
policy is post-match only; this change adds no forecasting feature or shot
model. Native validation adapts selected Pandera documentation on
[optional and nullable columns](https://pandera.readthedocs.io/en/stable/dataframe_schemas.html)
and [checks that retain nulls and compare columns](https://pandera.readthedocs.io/en/stable/checks.html).
Pandera is not installed or executed.

### Chronological experiments

`ModelTrainer` first reserves the final chronological holdout, keeping shared
kickoff buckets together. Its expanding-window quality checks use only the
earlier development set. A separate model then evaluates the holdout; an
all-history refit follows only after the quality gates pass. New bundles record
`evaluation_protocol=development_backtest_then_temporal_holdout_v1`. Existing
bundles and historical reports retain their original evaluations.

Within each XGBoost fit, early-stopping class weights come from its training
prefix. The ordered validation suffix uses unweighted log loss, matching the
outer probability metric. The subsequent refit uses class weights from all rows
provided to that fit. The [temporal audit](results/temporal-evaluation.json)
records a fixed 1,140-match PL cohort and all four previous/current protocol
combinations. Default quality gates rejected all four. Its two holdout fits are
diagnostics and do not override those rejections. Boundary regressions run with
`python -m unittest tests.test_model_pipeline.TestModelTrainer tests.test_core.TestXGBoostClassifier -v`.

Install the optional packages in the same virtual environment. The base Docker image includes XGBoost and public sources; install these extras to use the following commands there.

```bash
python -m pip install -r requirements-experiments.txt
python -m app.research analyze --source football-csv --competitions PL --seasons 2022,2023,2024 --backend compare --output .tmp/pl-analysis.json
python -m app.research experiment --source football-csv --competitions PL --seasons 2022,2023,2024 --trials 8 --output .tmp/pl-experiments.json
```

`analyze` reports team points, goals, clean sheets and points per match, plus outcome counts and median total goals. `--backend fpstreams` uses fpstreams grouping and aggregation; `compare` checks its results against Python and records a single aggregation timing for each. Download, normalization and imports are excluded. These retrospective totals are descriptive statistics and are never fed into prediction features.

`experiment` compares `xgboost,sgd,random_forest,gaussian_nb`. SGD trains logistic classifiers with stochastic gradient descent. Bayesian search uses a Gaussian process and expected improvement through scikit-optimize. Genetic search uses elite selection, crossover and mutation. These searches optimize model hyperparameters, rather than acting as standalone predictors. Use `--models sgd,xgboost`, `--optimizers bayesian,genetic`, `--trials 12` and `--seed 42` to choose the experiment. Each search uses exactly the requested trial budget, which must be at least four.

Additional models are `logistic_l1`, `logistic_l2` and `logistic_elasticnet`.
They fit multinomial logistic classifiers with SAGA for L1/Elastic Net and L-BFGS
for L2. Their scaler is fitted only on the training partition. L1 provides the
sparse classification analogue of Lasso. Outcomes remain categorical labels.
Search tunes the
inverse regularization strength `C`, plus `l1_ratio` for Elastic Net. The same
models are accepted by `train`, `all` and `compare-history`. Fits that fail to
converge raise an explicit error. L1/L2 mixing ratios are fixed by the model name.
Logistic fits use the library's standard `tol=1e-4` with a 20,000-iteration cap.

`--optimizers optuna` adds Optuna 5.0.0 with independent TPE, sequential trials
and in-memory storage. It uses the same parameter bounds and development loss
as the existing searches. TPE starts after `min(4, trials - 1)` successful random
startup trials; failed trials do not advance that count. There is no pruning or remote
study service. The optional requirements pin the tested version. See the
[Optuna sampler API](https://optuna.readthedocs.io/en/v5.0.0/reference/samplers/generated/optuna.samplers.TPESampler.html)
and [reproducibility guidance](https://optuna.readthedocs.io/en/v5.0.0/faq.html#how-can-i-obtain-reproducible-optimization-results).

```bash
python -m app.research experiment --source football-csv --competitions PL --seasons 2022,2023,2024 --models xgboost,logistic_l1,logistic_l2,logistic_elasticnet --optimizers bayesian,genetic,optuna --trials 8 --seed 42 --output .tmp/regularized-search.json
```

The `experiment` command tunes estimator parameters with fixed features. The
separate `feature-experiment` command below also searches recent-match and Elo
settings. Equal trial counts do not imply equal runtime or establish a general
optimizer ranking.

The trial budget counts attempted parameter configurations. Each configuration
uses up to three development-fold fits. Trials that fail to converge retain their
parameters and error with `log_loss=null`, and cannot be selected. Optuna records
its native failed state. GP and genetic search use a finite penalty of at least
36 to guide later proposals; this penalty is never reported as a valid model
score. Reports count completed and failed trials. If other configurations remain
valid, a search with no converged trial remains in the report as failed. A run
with no converged configuration fails explicitly. Invalid inputs and objectives
that are not finite still raise errors. Selection stays locked if the final development refit fails,
and that candidate has an error in place of holdout metrics.

The last roughly 20% of matches form a final holdout. Three expanding windows within the earlier data score each candidate by log loss. Matches sharing a kickoff stay in the same partition, and scaling is fitted on training rows only. The selected candidate is locked before holdout evaluation. JSON includes every trial, parameters, holdout accuracy, log loss, Brier score, RPS and majority/Elo accuracy baselines. Compare holdout results cautiously: choosing a new winner after reading them would reuse the test set.

### Joint feature and model search

```bash
python -m app.research feature-experiment --source football-csv --snapshots .tmp/history/football-csv --competitions PL --seasons 2022,2023,2024 --models xgboost,logistic_l2 --optimizer optuna --trials 12 --folds 3 --seed 11 --output .tmp/feature-search.json
```

This compares fixed defaults, model-only search and joint search with the same
attempt budget. `--optimizer` also accepts `bayesian` and `genetic`. Joint search
adds the last 2–12 matches, Elo update magnitude 5–50 on a log scale, and Elo
expectation home advantage 0–150. This window counts matches; it does not change
the number of training seasons. Model parameter bounds remain the existing ones.

Each feature recipe rebuilds its own development history. Search never receives
held-out matches, and all boundaries retain complete kickoff groups. Selection
uses three development folds by default and is fixed before held-out features
are built. The fitted model then stays fixed while pre-match history advances
after earlier kickoff results. Earlier kickoff alone does not prove that a match
has finished or its result has arrived, so this historical study does not verify
real-time availability.

The [study](results/feature-search.json) records five leagues, two models, three
fixed seeds and 12 attempts per search. This small budget in search spaces of
different dimensions cannot establish a general benefit from joint tuning.
The [Optuna search-space guide](https://optuna.readthedocs.io/en/v5.0.0/tutorial/10_key_features/002_configurations.html)
supports integer/log-scaled parameters and warns about added dimensions. The
later [Arntzen–Hvattum supplement, A.1](https://www.statmod.org/smij/Vol21/Iss5/Arntzen/Supplement.pdf)
informed the rating-parameter review. Its goal-margin update and initialization
procedure are not implemented here, nor is the original ordered-logit model.

The feature recipe specializes a fresh research instance of the existing
engineer. It cannot export a production bundle; default training and prediction
keep their existing settings. Aggregate reports omit individual forecasts and
match identities. Detailed CLI errors go to local stderr; aggregate JSON keeps
only their type and a general message. Private predictions used for the local
audit are retained outside the public inventory.

### Optional trees, rating variants and fixed-output explanations

Install the optional tree engines in the existing project environment:

```bash
python -m pip install -r requirements-tree-models.txt
```

The pinned engines accept the existing numerical features and emit A/D/H
probabilities. LightGBM uses deterministic CPU settings; GOSS changes its sampling
strategy. CatBoost uses Ordered boosting, `has_time=True`, chronological kickoff
validation, one training and prediction thread, and no training-file output.
No target-encoded team identifiers or categorical pipeline were added. Failed
refits invalidate the prior fitted state before input validation.

Successful default candidates and search trials retain every development fold's
fit/inference seconds and engine configuration. Trials also retain objective
seconds; searches retain total wall time, seed, requested and attempted trial
counts, and failed-trial history. Final fit and holdout inference are separate. Trials run
sequentially without pruning. Runtime does not affect selection, and equal trial
counts do not imply equal compute. Reproducibility checks exclude elapsed seconds
while preserving versions, thread policy, parameters, folds, and scores.

```bash
python -m app.research feature-experiment --source football-csv --snapshots .tmp/history/football-csv --competitions PL --seasons 2025 --models lightgbm,lightgbm_goss,catboost --optimizer genetic --trials 4 --folds 2 --output .tmp/tree-followup.json
python -m app.research rating-baselines --source football-csv --snapshots .tmp/history/football-csv --competitions PL --seasons 2025 --ratings basic,goal_weighted,pi --warmup-matches 3 --folds 2 --output .tmp/rating-followup.json
python -m app.research explain-model --source football-csv --snapshots .tmp/history/football-csv --competitions PL --seasons 2025 --class H --against A --max-rows 32 --output-dir .tmp/global-followup
```

These examples were exercised on an existing checksummed snapshot. They do not
refresh sources or save production models. Aggregate reports and four figure
types are in [the follow-up results](results/paper-followup-20261019).

Goal-weighted Elo uses `K=k*(1+abs(goal_difference))**power`; pi uses separate
home/away states, a signed goal-equivalent transform with c=3, logarithmic error
and cross-venue updates. These are explicit local controls. All configurations
share per-team match-count warmup and kickoff-group eligibility. Parameters are
selected on development folds before the final holdout. Rating versions, state
hashes and the separately fitted probability mapping are recorded. Earlier
kickoff results advance states while the mapping stays frozen. Kickoff does not
prove final-whistle or source-arrival availability. The ordered mapping does not
reproduce the pi paper's historical 28-bin mapping; neutral venues and lineups
remain unsupported.

`explain-model` fits the fixed default XGBoost on the development prefix and
explains the first bounded holdout rows, at most 200 rows and 50 features. Every
row uses the same output class or contrast. Contributions, baseline, and native
interactions close against separately predicted raw margins. Off-diagonal
interaction cells contain half the total pair effect. Figures show contribution
distributions, feature dependence in original units, and explicit interactions.
Private CSV exports retain all features and pairs with model/run identity,
source window and units. Public summaries omit individual inputs, contributions,
predictions, and match identities. These are tree-path explanations of raw scores,
not calibrated probability points, interventional effects or causal estimates.

The concentrated suite ran 372 tests: 369 passed and three optional checks were
skipped. The skipped SPADL/VAEP check passed separately in its existing isolated
socceraction 1.5.3 environment; two unchanged soccerdata wrapper checks remain
optional. Offline numerical audits and five CLI commands passed. Real Chrome
checks covered desktop/mobile views, keyboard table scrolling, legacy records,
numeric CSV exports and pitch SVG geometry. Production database access was
blocked by the task test runner. No commit, push or deployment was performed.

### Elo probability baselines

```bash
python -m app.research rating-baselines --source football-csv --snapshots .tmp/history/football-csv --competitions PL --seasons 2022,2023,2024 --ridge 0.001 --folds 3 --seed 42 --output .tmp/ordinal-elo.json
```

The four fixed configurations are ordered logistic regression with Elo only,
multinomial L2 logistic regression with Elo only, and the existing L2 and XGBoost
controls with all 30 features. Three expanding development folds select a model
before held-out features are built. Complete kickoff groups remain together.
During the holdout, model parameters stay fixed while earlier kickoff results
advance history. Match completion and result arrival times are unverified.

`elo_ordered_logit` uses `elo_diff / 400` with two ordered thresholds and no
extra intercept. It minimizes mean negative log-likelihood plus half the ridge
coefficient times the squared slope. The slope and first threshold lie in
[-30, 30]; the log threshold gap lies in [-12, 6]. These bounds and the default
ridge of 0.001 are local choices. Stable draw likelihoods, analytic gradients
and SciPy L-BFGS-B handle fitting; failed convergence raises an explicit error.

The estimator factory also accepts this model in existing experiments. Its
`ridge` search range is 1e-5 to 1 on a log scale. Joint feature search excludes
the unused recent-form window, retaining Elo update magnitude and expectation
home advantage. The default rating-baselines command retains the fixed basic Elo settings.
Optional goal-weighted and pi variants use the development-only protocol below.

The [study](results/ordinal-elo.json) records all 20 configurations across five
leagues and independently checks 20 ordered fits with direct thresholds and
SLSQP numerical derivatives. Full-feature controls exactly preserve the earlier
frozen forecasts. Metrics use previously reused historical matches; scaling,
regularization, and feature counts also differ across configurations. These
comparisons do not establish a causal model-family effect or future advantage.

The [official OrderedModel documentation](https://www.statsmodels.org/stable/generated/statsmodels.miscmodels.ordinal_model.OrderedModel.html)
supplies the cumulative probability formula and identifiability constraint.
Statsmodels itself is neither installed nor executed. The original 2010 paper
was available only through a publisher preview, so this native baseline is not
a reproduction. Goal-margin Elo updates, performance-rating initialization and
player ratings remain unimplemented. XGBoost remains the production default. The CLI
writes aggregates without individual forecasts or match identities.

### Calibration diagnostic sensitivity

```bash
python -m app.research calibration-diagnostics --predictions .tmp/fixed-probabilities.npz --bins 5,10,20 --minimum-bin-count 10 --output .tmp/calibration-diagnostics.json
```

The numeric NPZ contains `targets` (N integer labels), `probabilities` (M × N × 3),
and `model_names` (M unique, nonempty Unicode strings). Classes remain away win,
draw, home win. Pickle loading is disabled. The command fits nothing and writes
aggregate diagnostics without match identities or individual forecasts.

Joint TV groups the complete probability vector using all three coordinates
of `floor(B * p)`, clamping probability 1 into the last interval. These cube-grid
cells intersect the probability simplex. B is an axis resolution, not the total
cell count. Every occupied cell contributes its match-weighted total variation
between mean prediction and empirical class frequencies.

Scalar diagnostics use equal-width bins and approximate equal-mass bins. The
latter choose nearest cumulative-count boundaries between distinct scores,
using predictions only. Ties stay together, which can reduce the number of bins.
Classwise ECE averages the three scalar absolute gaps; top-label ECE uses the
first class in an exact maximum-probability tie. These describe different aspects
of calibration from joint TV and should not be ranked against one another.

`--minimum-bin-count` suppresses small cells from displayed tables only. Their
observations still contribute to ECE and reported suppressed-row counts. Empty
bins contribute zero. The existing `calibrate` command appends compact summaries
while retaining its original log-loss selection and other metrics.

The [study](results/calibration-diagnostics.json) preserves all 30 frozen maps
from five leagues and two base models. Its 1,069 unique holdout matches are
reused across models and maps. The [figure](results/calibration-diagnostics.svg)
shows the predeclared PL/XGBoost case and binning sensitivity. Reproduce it with:

```bash
python -m app.analytics.visualization.calibration --report results/calibration-diagnostics.json --output .tmp/calibration-figures
```

Selected [Vaicenavicius et al. methods](https://proceedings.mlr.press/v89/vaicenavicius19a/vaicenavicius19a.pdf)
motivate joint diagnostics and cautious interpretation. Binning bias and sample
noise remain. This local grid/cut implementation supplies descriptive estimates;
it does not reproduce the paper's experiments, hypothesis tests, resampling, or
proofs. No population calibration certificate, finite-sample lower bound,
p-value, confidence interval or production replacement is claimed.

### Draw probability calibration

```bash
python -m app.research calibrate --source football-csv --competitions PL,FL1,BL1,SA,PD --seasons 2022,2023,2024 --model xgboost --calibrators identity,temperature,dirichlet_l2,draw_constant,draw_logistic,draw_beta --snapshots .tmp/history/football-csv --output .tmp/draw-calibration.json
```

Three optional maps calibrate draw versus non-draw. `draw_constant` uses the
calibration suffix's draw frequency. `draw_logistic` fits a sigmoid to the draw
score. `draw_beta` uses log-score and negative log-complement features from
[Kull et al., Section 3 and Algorithm 2](https://proceedings.mlr.press/v54/kull17a/kull17a.pdf).
The two fitted maps use nonnegative coefficients and a fixed coefficient ridge
of 0.001 on mean binary log loss, with an unpenalized intercept. This ridge is
a local adaptation. All three maps require both binary outcomes in their
calibration window, including the constant control for comparability.

The output preserves the original away/home split conditional on a non-draw.
If its original mass is zero, that split is equal. This coupling is a local
football adaptation, without a joint multiclass calibration guarantee. Draw
log loss, binary Brier score, prevalence, mean probability and ten-bin ECE are
reported separately from the existing three-class scores.

Experiments that explicitly request a draw map use a numerical tie band of
1e-9 on development log loss. Priority is identity, draw constant, temperature,
draw logistic, draw beta, then Dirichlet. Final holdout labels do not select the
method. Original three-map-only calls retain their defaults and exact-tie
selection rule. Fitted maps pass finite-value, feasibility and projected-gradient
checks; the report retains the optimizer's status even if a line-search warning
occurs at a stationary solution.

The [study](results/draw-calibration.json) reuses 5,330 historical fixtures and
1,069 unique holdout matches across two base models and six maps. It records
60 candidate evaluations, 40 base fits and 240 map passes, including identity
and closed-form controls. All 30 earlier holdout probability arrays remain
exactly equal. Independent SLSQP fits with numerical derivatives verify 80 draw
maps, with 40 constant-window checks and 480 scalar holdout score checks.
The cohort has been used in earlier studies. An initial five-map comparison,
including its holdout scores, was inspected before adding the constant control
and tie rule. The final configuration is exploratory and does not establish
future performance. No data download, production model or website is changed.

### Probability calibration

The `calibrate` command compares raw probabilities (`identity`), temperature
scaling and `dirichlet_l2`. It can read existing checksummed season snapshots:

```bash
python -m app.research calibrate --source football-csv --competitions PL,FL1,BL1,SA,PD --seasons 2022,2023,2024 --model xgboost --snapshots .tmp/history/football-csv --output .tmp/match-calibration.json
```

Omit `--snapshots` to fetch the selected source. Offline snapshots must match
the requested source, league, and season. Each development fold reserves the
last roughly 20% of its training prefix for calibration and scores later games.
All boundaries preserve complete kickoff groups. Pooled development log loss
selects the map before the final holdout is scored. Identity and learned maps
share one base fit. The final fit keeps a separate calibration suffix; it does
not refit the base model on all development rows after calibration.

[Temperature scaling](https://proceedings.mlr.press/v70/guo17a.html) fits one
positive temperature to calibration log loss. This implementation solves the
monotone derivative in inverse temperature, with `T` bounded to `[0.01,100]`
and probabilities floored at `1e-15` before the log transform.
[Dirichlet calibration](https://arxiv.org/abs/1910.12656) uses multinomial
logistic regression on log probabilities. Here `C=1` applies L2 to coefficients,
with an unpenalized intercept; this is not the paper's ODIR scheme. Dirichlet
requires all three classes in its calibration window. The report retains failed
fits. A failed final calibration fit does not trigger selection of another map.

The optional `dirichlet_odir` map implements the paper's normalized ODIR
objective on log probabilities. It leaves diagonal coefficients unpenalized and
penalizes off-diagonal coefficients and intercepts separately, with fixed local
controls `lambda=mu=0.001`. Enable it explicitly with
`--calibrators identity,temperature,dirichlet_l2,dirichlet_odir`.
It uses the existing chronological windows and requires all three outcomes.
The default three maps remain unchanged. Independent objective, finite-difference
gradient and BFGS checks passed, including missing-class and degenerate windows.
The offline PL 2025/26 run includes this candidate. It does not reproduce the paper's internal
hyperparameter search, prediction ensemble, or original benchmark.

Reports retain every map, proper scores, classwise reliability and 9/10/11-bin
ECE sensitivity. No production bundle is saved. The historical
[local comparison](results/match-calibration.json) uses fixed default base
parameters and previously used snapshots, so it does not establish future
performance or the best joint model/calibration configuration.

Joint histogram diagnostics now also report an empirical Brier decomposition.
Replacing each forecast by its joint-bin mean gives
`binned_forecast_brier = uncertainty - resolution + reliability`.
The raw score adds `within_bin_remainder`, which can have either sign.
All occupied bins contribute, including cells hidden by the display threshold.
The class-summed score uses the existing three-class scale of 0–2. Outcome
frequencies describe the evaluated cohort and are not a prior forecast.
These sample- and bin-dependent components do not certify population calibration
or select a model. Independent bin-mean forecasts verify the identity, both
remainder signs, class permutations, display thresholds and legacy inputs.
Previously frozen reports and figures retain their original versions.

The [pinned scikit-learn source](https://github.com/scikit-learn/scikit-learn/tree/85593ca7a07a9956ac891f6bfc3fedca53b74f72)
was read without executing or installing that revision. Its binary Brier scaling,
bin-edge assignment and percentile cuts differ from some native diagnostics.
Its log-loss clipping uses dtype epsilon rather than the local `1e-15` floor;
its sigmoid path transforms `predict_proba` to logits, whereas `draw_logistic`
uses raw draw probabilities. The pinned main branch and the earlier stable API
review describe different versions; neither changes the project dependencies.

By default, `score-models` compares three fixed score configurations on the same chronological
partitions: `poisson`, `poisson_decay` and `dixon_coles`. It uses NumPy and SciPy
from the main requirements; optional experiment packages are unnecessary.

```bash
python -m app.research score-models --source football-csv --competitions PL,FL1,BL1,SA,PD --seasons 2022,2023,2024 --snapshots .tmp/history/football-csv --output .tmp/score-models.json
```

Snapshot checks enforce source, league, season, and checksum identity. Omit
`--snapshots` to fetch data. `--models` selects a subset; `--half-life-days 365`,
`--ridge 0.01`, `--folds 3` and `--max-iter 1000` are the defaults. Static Poisson
ignores the half-life; zero disables decay in the other configurations too.
Decay is measured from the last training kickoff in each fit. The roughly 20%
holdout and development folds preserve complete kickoff groups. Pooled
development log loss locks the selected configuration before holdout scoring.
Incomplete development fits are ineligible; a failed final fit remains failed
without selecting a replacement.

The independent-component rates combine centered log attack and defensive weakness, an intercept,
and a home effect. The joint Poisson likelihood follows the model family in
[Maher](https://doi.org/10.1111/j.1467-9574.1982.tb00782.x).
[Dixon–Coles](https://research.lancaster-university.uk/en/publications/modelling-association-football-scores-and-inefficiencies-in-the-f/)
adds a correction to 0:0, 0:1, 1:0 and 1:1, with exponential time weights. The
local estimator adds ridge regularization, bounded rate parameters and a
conservative rho envelope valid for every fitted and neutral unknown-team pair.
Ridge penalizes raw attack/defence coordinates in the normalized weighted
likelihood; these settings do not reproduce the original unpenalized estimator.
The optimizer uses an analytic gradient and one deterministic initialization;
convergence does not certify a global likelihood optimum.

Unknown teams receive zero log attack and defence, the fitted geometric-average
strength. They are excluded from the training team universe until observed.
Parameters stay fixed throughout each evaluation window. Outcome probabilities
use the full-support Skellam distribution and exact four-cell correction, without
normalizing a truncated score grid. Reports include outcome log loss, Brier,
RPS, reliability, 9/10/11-bin ECE, exact-score log loss, goals MAE and unseen-team
counts. Score likelihood and outcome loss can favor different configurations.

Future model summaries now state the `away_win, draw, home_win` class order,
the infinite-support outcome calculation and the time-weighting convention.
A half-life of `h` elapsed 24-hour days corresponds to `xi=log(2)/h`; zero
disables decay. Fitting weights sum to one before the ridge objective.
The score model always applies its home effect to the listed home team and
does not consume neutral-venue flags. The summary and probability checks passed;
the explanatory fields do not change fitting or prediction calculations.

The optional `poisson_constant` control fixes all team log strengths at zero
and fits only the common home/away rates on each training prefix. It shares the
native intercept and home-effect bounds, so it is a bounded adaptation of the
intercept-only hierarchy rather than the paper's unrestricted estimator. The
default three configurations remain unchanged; the research factory uses no
time decay for either `poisson` or `poisson_constant`. Inactive rho coordinates
are now reported as `null` in future summaries.

The optional `bivariate_poisson` candidate uses independent counts U and V plus
a common Poisson count C: home=U+C and away=V+C. The log predictors give the
U/V rates; marginal expected goals add `shared_goal_rate`. Outcome Skellam
probabilities use the U/V rates because C cancels from the difference. Exact
score likelihoods sum all feasible C counts in log space. The local analytic
gradient includes the one-sided derivative at zero shared rate; the optimizer
fits a common nonnegative rate bounded at 3. This bound, fixed ridge and one
deterministic initialization are local choices. They do not reproduce the
original EM estimator, diagonal inflation or a dynamic state-space model.
The research factory disables decay for this static candidate. New summaries
explicitly identify point estimation and the absence of posterior intervals.
Independent convolution, 60-digit scalar masses, finite-difference gradients,
zero and interior shared-rate fits, unseen teams, convergence failures, and
holdout isolation passed. The offline PL run includes this candidate.

Each validation fold and final holdout also gets `goal_count_diagnostics`:
home/away counts for 0, 1, 2, 3 and 4+ goals, plus joint 0:0, 0:1, 1:0, 1:1
and other scores. Expected counts sum fixture probabilities, with the exact DC
correction retained in the joint cells. Bivariate Poisson uses total means for
marginals and the shared-component joint law for low scores. Tail bins keep the remaining mass;
displayed cells are not renormalized. These are descriptive point-fit counts,
without posterior intervals, goodness-of-fit p-values or selection by fit to
these counts. The pooled development selection still uses outcome log loss.

The opt-in workflow below was also exercised on the PL 2025/26 snapshot.
The example retains its earlier multi-season inputs; PNG/SVG output and rejection
of missing or inconsistent count panels were checked:

```bash
python -m app.research score-models --source football-csv --competitions PL --seasons 2022,2023,2024 --snapshots .tmp/history/football-csv --models poisson_constant,poisson,poisson_decay,dixon_coles,bivariate_poisson --output .tmp/score-models-counts.json
python -m app.analytics.visualization.score_counts --report .tmp/score-models-counts.json --competition PL --output .tmp/score-count-figures
```

The renderer overlays expected-count markers on observed-count bars and writes
PNG/SVG using the existing figure style. It requires the new diagnostic fields
and rejects inconsistent totals or observed panels. It cannot generate missing
numbers from an old report. Historical score results and figures are unchanged.

The original Dixon–Coles decay rate of 0.0065 is measured per half-week.
The local interface uses elapsed 24-hour days and a fixed default half-life;
that value does not reproduce the original decay selection, rolling refits,
cross-division estimation, parameter standard errors or bookmaker-return study.

The [Karlis–Ntzoufras article](http://www2.stat-athens.aueb.gr/~jbn/papers2/08_Karlis_Ntzoufras_2003_RSSD.pdf)
was read in full. Its historical fit tests and draw inflation are not current
forecasting evidence. The [Baio–Blangiardo manuscript](https://discovery.ucl.ac.uk/id/eprint/16040/1/16040.pdf)
was also read in full, including its code appendix and explicit one-batch
in-sample limitation. Fixed ridge is not a learned hierarchical prior, and
point-fit frequencies are not posterior predictive validation. The separate
[2012 Koopman–Lit preprint](https://research.vu.nl/ws/files/3167073/12099.pdf)
has been read in full; the 2015 published article and official online appendix
remain unread after their download endpoints returned 403. Its state-space
smoothing, rolling refits and historical betting illustration were not reproduced.

The [2017 LightGBM paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)
and its five-page official supplement have been read in full. GOSS reweighting,
EFB exclusivity and the original hardware/data scale bound the interpretation
of its benchmarks. Optional LightGBM, GOSS and CatBoost adapters now run with
shared chronological partitions and search budgets. Local stage timings are
recorded; they do not establish a general speed ranking. XGBoost v3 (13 pages, including its weighted-sketch
appendix) and CatBoost v5 (23 pages, including appendices A–G) have now been read
in full. Neither historical systems benchmarks nor the CatBoost iid analysis
establish local runtime or football probability guarantees.

Selected release-source reads pin LightGBM 4.7.0 at
`8f7036f03627054d5a54a6f965b13f4b9ff2cb63` and CatBoost 1.2.10 at
`b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084`. LightGBM's sampling strategy,
CPU determinism and classifier probability paths were inspected. CatBoost's
`has_time`, `boosting_type`, disabled file-output option and classification paths
were inspected; its probability-prediction call has a separate thread-count
argument. Source reading preceded implementation. The project environment now
contains LightGBM 4.7.0 and CatBoost 1.2.10 alongside the existing `xgboost-cpu` 3.4.1.
Real temporal searches, failed-refit invalidation and stage-cost exports passed
in the concentrated verification phase.

Two existing visualization references have now been read in full: the
[Heer–Bostock CHI paper](https://hci.stanford.edu/publications/2010/crowd-perception/heer-chi2010.pdf)
(10 pages) and the archived v3 of
[Task-Based Effectiveness of Basic Visualizations](https://arxiv.org/pdf/1709.08546)
(9 pages). Static source review retains the current shared-axis comparisons,
direct labels and lookup tables. The papers do not prescribe one chart for
every task, universal pixel/alpha settings or interaction guarantees. Current
browser, narrow-screen and export checks remain deferred; no original user
study or external study repository was reproduced.

The [five-league study](results/score-models.json) includes frozen default
XGBoost/L2 comparisons from the regularized experiment after matching identities
and labels. They do not participate in score-model selection and use different
feature and class-weight settings. The study adds provenance and verification
to the CLI output; private prediction arrays remain local. It uses previously
studied 2022/23–2024/25 snapshots and fixed hyperparameters. It saves no production
bundle and does not establish future performance.

The calibration report also retains a metadata correction: the imported
`xgboost` module belongs to the `xgboost-cpu` distribution. Version 3.4.1 is
confirmed by both runtime and distribution metadata. The revised report keeps
its original execution ID and unchanged numerical payload.

`compare-outcomes` reads paired fixed forecasts from a local NPZ, without
refitting or selecting models. It requires the main runtime, including XGBoost
through the shared probability validator. CLI help remains lazy.

| NPZ field | Shape and meaning |
| --- | --- |
| `targets` | `(n,)` integer labels, `0=L, 1=D, 2=W` from the home team's perspective |
| `match_ids` | `(n,)` unique positive integer match identities |
| `kickoffs` | `(n,)` Unicode ISO timestamps with explicit time zones |
| `model_names` | `(m,)` unique nonempty Unicode names, at least two |
| `probabilities` | `(m,n,3)` numeric probabilities in `L,D,W` order, aligned to those same rows |

Prepare the NPZ from your existing local forecasts. Object arrays are rejected
with `allow_pickle=False`. IDs, labels, and forecasts remain local; the report
contains aggregate scores, date coverage, counts, and identity digests.

```bash
python -m app.research compare-outcomes --predictions .tmp/forecast-comparison-20261017/PL-paired-private.npz --reference xgboost/default --output .tmp/paired-comparison.json
python -m app.analytics.visualization.outcomes --report results/outcome-comparison.json --output results
```

Rows are sorted by UTC kickoff and identity, then grouped by UTC date. Probability
rows retain their original values within the evaluator's `1e-6` mass tolerance,
including float32 values upcast into a shared float64 array. The report records
the largest row-sum error. Only log loss uses a `1e-15` floor; Brier and RPS use
the original values. Differences are candidate minus reference, so lower is better.

Defaults use `--replicates 2000`, `--seed 20261017`, `--block-lengths 1,3,7`,
and `--hac-lags 0,1,3,7`. Every method and metric shares paired draws. A block
continues to the next observed match-day with probability `1-1/L`, otherwise
restarts uniformly; the final observed day wraps to the first. Each replicate
pools whole-day loss totals divided by its sampled match count. Unequal daily
counts are preserved. These are observed days, not equally spaced calendar days.
At least two distinct UTC dates are required, block lengths cannot exceed their
count, and HAC lags must be smaller. The default lag grid therefore needs eight
dates. Invalid settings fail rather than silently changing the grid.

For day `g`, let `S_g` be its paired loss total, `n_g` its match count, `N` the
total matches, and `d=sum(S_g)/N`. The Bartlett diagnostic uses the ratio
influence `u_g=S_g-n_g*d`. Its variance is
`[sum(u_g^2)+2*sum_{k=1}^q (1-k/(q+1))*sum_g u_g*u_(g-k)]/N^2`.
With one match per day this reduces to the usual sample-mean HAC variance.
This is a grouped adaptation, with no finite-sample correction or DM p-value.

The new `day_influence_acf` field describes these same influences for every lag
from zero through the maximum requested HAC lag, using
`sum_g u_g*u_(g-k)/sum_g u_g^2`. Zero variance produces `null` correlations.
Lags count observed days, including across gaps; the output has no significance
bands, automatic lag selection or stationarity conclusion. Independent day
aggregation and dense HAC checks passed, including negative correlation and
zero variance. Frozen historical comparison reports and figures are unchanged.

The [frozen comparison](results/outcome-comparison.json) adds source receipts and
independent verification to five CLI outputs. The selected score configurations
remain those chosen by development folds in the earlier study. Its pointwise
bootstrap percentiles and lag sensitivity are exploratory diagnostics. They
condition on fitted models and previous selection, assume weak dependence across
observed days, and do not preserve calendar gaps or season breaks. They do not
establish simultaneous coverage or future superiority. The SVG/PNG renderer uses
the existing figure style and requires Matplotlib from the scouting requirements.

The match `experiment` command does not save or publish models. To train a chosen candidate, pass its parameters to the existing trainer; the calibration, stability and baseline gates still apply:

```bash
python -m app.pipeline train --source football-csv --competitions PL --seasons 2022,2023,2024 --model sgd --params '{"alpha": 0.1}'
```

Alternative models use the same `L,D,W` probability and bundle contracts. They report explanations as unavailable; XGBoost retains TreeSHAP explanations. Small datasets do not imply a speedup from fpstreams, and a successful hyperparameter search does not imply better future predictions.

## Multi-season and cross-league comparisons

Collect public history once, then run comparisons without network access:

```bash
python -m app.research collect-history --seasons 2016,2017,2018,2019,2020,2021,2022,2023,2024,2025 --output .tmp/history-collection.json
python -m app.research compare-history --test-seasons 2023,2024,2025 --windows 1,3,all --output .tmp/history-comparison.json
```

Both commands default to registered public leagues and both public sources. Use `--competitions PL,BL1`, `--sources football-csv` and `--cache-dir .tmp/history` to limit the scope. Collection caches one JSON snapshot per provider, competition and season, with source URLs, retrieval time and a SHA-256 digest. Existing snapshots are verified and reused. `--refresh` deliberately replaces them with current upstream data. `manifest.json` lists the requested collection, counts and failures. Missing files or failed downloads are errors, rather than empty successful seasons. OpenFootball does not provide every historical league-season; CSV coverage is broader in this collection.

`compare-history` trains on the previous one season, three seasons, or all available earlier seasons and evaluates each entire requested test season. It compares local league models with models pooled across leagues from the same provider. League indicators let pooled models distinguish competitions. Providers are never pooled together. Each pooled fit ends before the earliest held-out fixture across its participating leagues, and local models use that same cutoff. Earlier results within the test season update rolling team/Elo features for later fixtures; the estimator stays fixed throughout the test season.

These experiments group matches by their source calendar date, so providers with missing kickoff times share a consistent daily evaluation granularity. Explicit team-name aliases preserve experimental history across common provider naming changes. Ambiguous changes such as Belenenses versus Belenenses SAD stay separate. These transformations apply to experiment copies and do not change production identities or cached source records.

The regular `train` command reads its selected provider directly and does not apply this cached cross-provider audit. Review the audit before using older provider histories for production training.

JSON reports source coverage, shared matches, unmatched examples and conflicting dates or scores, plus model metrics for every league, test season, history window and local/pooled strategy. Date comparisons use unique home/away pairs within a season, so duplicate pairings such as playoff rematches are not forced into a date match. Records with cross-provider date or score conflicts are isolated from both providers' experiment copies; cached originals remain intact. Reference providers already present in the cache are used for this check even when only one provider is selected for modeling.

Test-population hashes verify that paired window/strategy comparisons use the same matches. Aggregate metrics are weighted by test-match count and include variation across league-seasons. `paired_comparisons` reports improvements and regressions relative to one-season local training, giving each paired league-season equal weight. Provider comparisons can have different coverage; they are not automatically paired. All-history results list missing earlier seasons explicitly.

The matrix uses fixed model settings, including SGD `alpha=0.1` from the earlier PL exploration, so its results are retrospective rather than fresh independent validation. Use `--models xgboost,sgd`, `--windows 1,all` or `--no-pool` for smaller comparisons. A `.partial.json` checkpoint is updated during a run. Collection failures remain visible in the final report and produce a nonzero exit even when all available model comparisons succeed. More history or more leagues can help, but older playing conditions and different league distributions can also reduce accuracy. These commands do not save or publish production models.

## Public scouting data

The scouting commands collect each selected provider's published inventory. They require no API key:

```bash
python -m pip install -r requirements-scouting.txt
python -m app.research collect-open-data --output .tmp/open-data-inventory.json
python -m app.research scout-events
python -m app.research scout-tracking
python -m app.research scout-bodypose
python -m app.research scout-figures
```

Use `--datasets football-csv,statsbomb` to limit collection, `--data-dir` to choose storage, and `--refresh` to fetch new upstream revisions. The complete collection is large: StatsBomb alone contains about 16GB of uncompressed JSON. Manifests record source revisions and per-file hashes. Git LFS entries are resolved to real payloads and checked against published hashes. Analytics verify asset hashes when reading their inputs.

| Dataset | Contents | Source |
| --- | --- | --- |
| Football-Data archives | Every published European season ZIP from 1993/94 through the current season; up to 22 divisions, match statistics, referee, attendance and selected pre-close/closing odds | [Official downloads](https://www.football-data.co.uk/downloadm.php) |
| OpenFootball | Every published season JSON, including leagues outside the European CSV collection | [OpenFootball](https://github.com/openfootball/football.json) |
| StatsBomb | Competition/season catalogs, matches, lineups, event coordinates, supplied shot xG and available 360 freeze frames | [StatsBomb Open Data](https://github.com/hudl/open-data) |
| Metrica | All three anonymous sample matches with synchronized events and continuous tracking, in CSV and FIFA EPTS formats | [Metrica Sports](https://github.com/metrica-sports/sample-data) |
| SkillCorner | All match folders present in the repository, broadcast tracking, dynamic/off-ball events, phases of play and supplied A-League season aggregates | [SkillCorner](https://github.com/SkillCorner/opendata) |
| SkillCorner body pose | Both full public 25FPS matches with 29 predicted joints per player and uncertainty estimates | [Body-pose data](https://huggingface.co/datasets/SkillCorner/opendata-bodypose) |

Provider readmes/catalogs can lag behind repository contents. Inventories follow the actual files and report what was obtained. Archive presence does not establish complete league coverage. The latest football season is partial, and some StatsBomb seasons focus on one club or player. Unreferenced event files stay in the raw archive but are excluded from chronological analysis because their match dates and competition metadata cannot be established.

Historical CSV collection/comparison now accepts `PL,ELC,EL1,EL2,ENL,SPL,SCC,SL1,SL2,BL1,BL2,SA,SB,PD,SD,FL1,FL2,DED,BPL,PPL,TSL,GSL`. Additional codes identify historical divisions and are not football-data.org subscription codes. Bulk snapshots are under `.tmp/open-data/football-csv/history`, usable with `compare-history --cache-dir`. Missing statistics remain null. Odds retain their column-defined stage; collection timestamps are unavailable and odds are not automatically forecasting features.

`scout-events` produces competition-season-team-player summaries, covered lineup exposure, progressive passes/carries, box entries, pressures, passing networks and 360 spatial context. Successful progression must reduce goal distance by at least ten StatsBomb units and 25%. Coordinates use the provider's 120x80 system, rather than actual stadium dimensions. Mean pass origins describe possession locations. Visible teammate width/depth and lane-blocker counts are event-time snapshot proxies.

The expected-threat grid is an empirical shot/move Markov model fitted retrospectively to the analyzed archive. Player net xT includes successful passes/carries and can be negative. Player rates use covered lineup intervals; unknown exposure stays null. Position labels come from observed event roles. Compare within competition-season, role and adequate coverage. The mixed public archive is not a representative transfer-market population.

The shot experiment compares supplied StatsBomb xG with logistic regression and histogram gradient boosting. It holds out whole matches by date, excludes shootouts and penalties, and uses shot-origin geometry, header/set-piece flags and recorded pressure. Outcomes, shot endpoints and provider xG are not model features. This is a pooled historical experiment. For a narrower opposition study:

```bash
python -m app.research scout-events --competition-id 55 --season-id 282 --output-dir .tmp/scouting/euro-2024
```

`scout-tracking` streams meter-based positions and reports detected-distance estimates, high-intensity/sprint segments, sample peak speed, observation coverage, occupation grids and team shape. Gaps over 0.25 seconds and speeds above 12m/s are excluded. Local high-intensity and sprint thresholds are 5.5 and 7m/s; these differ from some vendor definitions. Shape estimates require eight visible players, are sampled at most once per second, and exclude keepers when their role is known. Metrica CSV samples lack keeper labels, so their shape summaries include tracked keepers.

SkillCorner extrapolations do not count as observed physical distance. Supplied dynamic-event and season-aggregate metrics retain their definitions and QC/inclusion rules. Empty quality flags stay unknown. Average-performance fields are not season totals, and `p30tip` rates use 30 minutes in possession. Anonymous Metrica identities cannot be linked across games. Observation coverage and comparable conditions matter when interpreting workload or scouting differences.

Body-pose analysis runs on both full matches and handles both periods, substitutions, missing poses and ZIP resource metadata. It records pose coverage and sampled predicted joint uncertainty. Pose XY is interpolated from 10Hz tracking, so 25Hz pose frame numbers cannot be joined directly to tracking frame numbers. Joint Z is reliable relative to the player centroid, rather than necessarily to the pitch. The output characterizes model-estimated joints and their uncertainty.

Outputs are under `.tmp/scouting`: JSON reports, CSV player summaries, match drilldowns and PNG/SVG pitch figures. Attribution and official source links accompany the analyses. A corrupt optional 360 JSON is isolated while verified event/lineup data stays usable. Core failures and optional exclusions are recorded separately, and raw assets remain unchanged.

## Unified football database

The self-contained SQLite database is `data/football.sqlite3`. It stores original
files as compressed chunks with SHA-256 digests, plus CSV tables, JSON documents,
tracking frames, pose frames and native SQL views. Use SQLite to query its tables and views without Python or the original downloads.

```bash
python -m app.data_service.ratings --output .tmp/open-data/game-ratings
python -m app.data_service.build_warehouse \
  --database data/football.sqlite3 --sources .tmp/open-data \
  --fm-root /path/to/decoded-fm/data
```

The optional FM directory contains versioned `csv_*` folders and their conversion
reports. All decoded tables and available original archives are imported. Imports
can be rerun: completed assets and tables are reused, while changed file contents
are stored as a separate revision. The generated `football.inventory.json` lists
sources, imported row counts and quarantined documents.

To collect recent ESPN observations and current-season Football-Data CSVs, run:

```bash
python -m app.data_service.recent_matches --start 2026-10-13 --end 2026-10-15 --collect-only
```

Each run prints a new snapshot directory. Use `--root PATH --resume` with the
same date range to continue an interrupted collection. Resume reuses verified
responses; a fresh collection needs a new directory. The receipt records network
attempts, downloads, cache hits and the range of source observation times. Cached
responses keep their original times, and ingesting an older snapshot preserves
newer match observations and their summaries. To import a reviewed snapshot, use
`--root PATH --ingest-only` and point `--international-root` at an existing
verified international-results archive. Older collection manifests remain
ingestible; collecting into them requires a new directory.

Summary capture times are checked independently of scoreboard times. A result
revision without a matching replacement removes the old summary link while
retaining its archived bytes. An additive `recent_source_observation_times`
table records verified capture times; the four-column summary table stays
compatible with existing readers and writers. Equal or older captures cannot
resolve a conflicting fixture. Legacy conflicts without a verified capture
time remain excluded pending review.

The general builder routes `soccerdata` through its dedicated importer: public
frames, schemas, receipts and caches are retained; browser profiles and runtime
files are excluded. Source-family aliases and hard-linked source files are
rejected. Use independent files in collection directories. Wyscout downloads
accept plain filenames and verify downloaded bytes before replacing a destination.

Useful tables/views: `datasets`, `assets`, `csv_columns`, `csv_irregular_rows`,
`repository_files`, `soccerdata_imports`,
`documents`, `football_matches`, `football_results_raw`, `statsbomb_matches`,
`statsbomb_events`, `statsbomb_visible_players`, `skillcorner_tracking`,
`tracking_positions`, `pose_joints`, `fm_players`, `fm_player_ratings`, `fm_player_history`,
`fm_employment_history`, `fm17_players`, `fifa_male_players`,
`fifa_female_players`, `fc24_male_players`, `fc25_all_players`,
`fc26_ea_fc26_players`, `game_edition_coverage`, `game_profile_versions`.
Recent research also provides `match_coach_observations`,
`coach_tactical_team_matches`, `tactical_evidence_claims` and
`tactical_switch_windows`, with original match/event identifiers retained.
High-volume JSON views expand stored frames on demand; filter by asset or match
before expanding pose joints. Raw files can be recovered with
`Warehouse.read_raw(asset_id)`; for large assets use `python -m app.data_service.database --export-asset ID --output /path/to/file`.

```sql
-- Compare ratings only within the same game and dated roster update.
SELECT fifa_version, fifa_update_date, overall, potential, club_name
FROM fifa_male_players WHERE player_id = '158023'
ORDER BY fifa_version, fifa_update_date;

-- FM snapshots retain native ability and potential codes.
SELECT d.edition, p.person_id, p.full_name, p.current_ability, p.potential_ability
FROM fm_players p JOIN assets a ON a.id = p.asset_id
JOIN datasets d ON d.id = a.dataset_id WHERE p.person_id = '735216';

SELECT frame, player_id, joint_name, x, y, z, p90_mae_cm
FROM pose_joints WHERE match_id = '1925299' AND frame = 50000;
```

Public FM UI exports retain their original formatting and unknown values. Exports
without game UIDs use explicit local row IDs; percentage role ratings are not CA.
Query irregular CSV rows in `csv_irregular_rows`; the inventory records their
count, and the original file remains available in raw chunks.
Provider player IDs remain separate. `conservative_identity_candidates` exposes
unique exact name/date-of-birth matches for review; it does not merge identities.
FM ability uses its native 0–200 scale and negative potential codes remain ranges,
not fixed scores. CSV source columns remain text to preserve their exact values; cast numeric fields
when filtering, or use `fm_player_ratings` / `game_profiles`. FIFA/EA ratings use their own scale and are game attributes,
not measured physical performance. FIFA roster update dates remain available to
avoid using future updates in historical experiments. The public EA endpoint's
edition/effective date is unverified because its response differs from the current
ratings website. FC25 exports retain source-encoded birth dates. Canonical birth dates are populated
only when the decoded Excel serial agrees with the EA birth date for the same ID;
conflicts stay missing in the canonical profile.
`python -m app.data_service.database --verify` checks SQLite integrity, foreign
keys, completed assets and sample file digests from each dataset.
Malformed upstream JSON is preserved in raw chunks and recorded in `quarantine`.

## Validate game ratings against real matches

```bash
python -m app.data_service.wyscout collect
python -m app.data_service.wyscout ingest --database data/football.sqlite3
python -m app.data_service.references --database data/football.sqlite3
python -m app.analytics.cross_validation --backend compare
python -m app.analytics.event_validation
python -m app.analytics.validation_figures
```

`cross_validation` links SkillCorner players to dated FM/FIFA snapshots using
normalized full-name tokens and exact birthdays. It allows extra middle names,
requires a unique game ID and stable game birth date, and rejects initials-only
matches. Camera coverage and played minutes determine eligible observations.
Weighted physical/run aggregates are checked against fpstreams when using
`--backend compare`.

`event_validation` uses the public [Wyscout soccer-log archive](https://figshare.com/collections/Soccer_match_event_dataset/4415000)
and its [data paper](https://www.nature.com/articles/s41597-019-0247-7). It compares
historical FIFA attributes with passing and supplied PlayeRank performance across
five 2017/18 leagues. Game snapshots must precede the player's earliest selected
appearance date. Same-day updates are excluded when update time is unknown.
Analyses require adequate minutes and pass attempts, retain all four
predeclared hypotheses, and report player bootstrap intervals.

Associations adjust for position, age, team and observed pass context. Predictive
comparisons use position, age and team as the baseline, then add one game attribute;
outcome-derived pass geometry and exposure controls are excluded from predictions.
Player holdouts and leave-one-league-out tests use the same rows for both models.
These retrospective tests remain exploratory and require future-season validation.

Query `real_players`, `real_observations`, `game_real_links` and
`cross_source_validation` for the underlying links and results. Reports are
`.tmp/cross-source-validation.json` and `.tmp/event-game-validation.json`; figures
are under `.tmp/scouting/validation`. A link's `evidence` records source names,
birthdays, game IDs and the historical date cutoff.

GitHub reference files are pinned by commit and checked against Git blob hashes.
They are stored as source artifacts and inspected statically. Role-weight formulas,
website attribute normalizers and FM simulation outcomes provide hypotheses;
their real-world validity requires separate evidence. The FM26 reference CSV with
50 headers and 116 fields per row stays in the raw archive and irregular-row table.
FMInside direct pages/update downloads currently return HTTP 403; the source-status
record preserves that gap and distinguishes community updates from native snapshots.

## Player and team assessment

Run the offline models after importing the Wyscout, historical CSV and optional game data:

```bash
python -m app.research assess --database data/football.sqlite3
python -m app.research assess --models ridge,bayesian_ridge,elastic_net,sgd --train-cap 5000
```

The player analysis uses five leagues from 2017/18. It builds exposure-aware
profiles, beta/gamma empirical shrinkage estimates and relative percentiles within
league and role. The manually specified role weights are recorded. Goalkeepers
have separate passing and recorded save outcomes. Their minutes are reconstructed
from lineups, substitutions and red cards, with no assumed stoppage time. Save
rates lack shot-difficulty adjustment.

Ten regressors compare next-window non-penalty goals, progression, pass completion,
duel wins and goalkeeper outcomes: Ridge, Bayesian Ridge, Elastic Net, SGD, SVR,
random forests, Extra Trees, histogram gradient boosting, XGBoost and MLP. Each
tries two fixed parameter settings. Validation selects both the algorithm and
baseline. The final time window stays held out. A second experiment excludes a
fixed 20% of player identities from training, validation and empirical priors.
Targets condition on sufficient future playing time and attempts. Matched-subset
ablations compare the same players with and without dated FIFA attributes.

KMeans, diagonal Gaussian mixtures, PCA and NMF describe playing styles. KMeans
uses held-out silhouette selection; GMM uses training BIC. Similar-player lists
compare historical players in the same broad role. The styles and percentiles
describe observed activity and are not current transfer recommendations.

Team models use eight past fixtures to forecast mean goals for, goals against and
points across the next five, within one league-season. The 2024 season selects
models and 2025 remains held out. Regularized Poisson models estimate team attack,
opponent defence and a home effect in each league, using recency weights with a
one-year half life. Dated deployment ratings are fitted separately after evaluation.
Their scales support comparisons within each league. Wyscout event styles describe
98 teams in the covered 2017/18 leagues.

Outputs include local JSON, test predictions and reusable model/style bundles
under `.tmp/assessments`. `assessment_runs`, `player_assessments`,
`team_assessments`, `assessment_experiments` and `assessment_examples` retain the
results in SQLite. Source fingerprints and cache content hashes invalidate stale
analysis caches. Production match models and dashboard recommendations use their
existing training and release rules.

## Formation, direction and opponents

```bash
python -m app.research tactics --database data/football.sqlite3 --train-cap 15000
```

This analysis reads date-referenced StatsBomb fixtures from the warehouse.
Starting XI and Tactical Shift events supply formation codes. Role locations are
means of on-ball actions. Possession uses event-clock and action-share proxies,
with clock coverage reported. Official possession percentages remain unknown.
Completed moves are grouped into low-y, central and high-y thirds in the provider's
attacking coordinate system. Opponent dossiers stay within competition and season
and retain home/away counts.

Chance models predict whether a later non-penalty shot occurs in the same
possession after the first three on-ball actions end. Prefixes with an earlier
shot are excluded. Features contain only information known by that decision clock.
Eight classifiers use the same temporal match partitions and fitting budget.
Logistic feature ablations test formation, channels/width and past opponent context
on the same rows. All tested results remain visible, including poor models.
Match-cluster bootstrap intervals describe conditional uncertainty.

Historical opponent context uses archive-conditioned Elo and up to eight earlier
covered matches, with same-day updates applied after predictions. Selective archive
coverage and long gaps limit that context. Descriptive formation/possession/xG
associations do not identify the effect of changing tactics.

Outputs are under `.tmp/tactical-relations`. SQLite retains `tactical_assessment_runs`,
`tactical_team_matches`, `tactical_opponent_dossiers` and `tactical_tracking_context`,
along with compressed artifacts.

## Rotations, transitions and external load

```bash
python -m app.research movement-lab --database data/football.sqlite3
python -m app.analytics.research_figures --output results --database data/football.sqlite3
```

The movement lab reads the 20 imported SkillCorner matches directly from SQLite.
It follows the contextual approach in FIFA's
[Croatia rotation analysis](https://www.fifatrainingcentre.com/en/game/game-analysis/world-class-tactics/croatia-evolving-third-player-combinations.php)
and [Boothroyd transition session](https://www.fifatrainingcentre.com/en/practice/elite-sessions/transition-to-attacking/transitions.php).
Five consecutive detected possession snapshots identify persistent pair-order
exchanges. Both players must move and remain nearby. These are spatial review
candidates; tactical intent and third-player combinations need further event or
video verification. The decision frame follows confirmation of the exchange.

Vendor `transition` and `quick_break` phases supply windows of three to ten seconds.
The analysis links observed player load, counter-press event labels and defensive
width/depth. Rest-defence counts also require a detected ball and eight detected
outfield players. Each team's attacking direction comes from period metadata.
Vendor classifications and extrapolated phase geometry retain their uncertainty.

The [UEFA EURO 2024 physical report](https://editorial.uefa.com/resources/0297-1d4e3592fbf1-f11d4e1c826a-1000/uefa_euro_2024_physical_analysis_report_20250318094958.pdf)
motivates possession/position context, speed zones above 20 and 25km/h, and rolling
one, three, five and ten-minute demands. This implementation uses consecutive
detected positions and half-second chord velocities. Gaps and raw speed spikes
reset the filter. It reports speed-threshold sensitivity, scalar acceleration,
deceleration, sprint bouts and repeated-sprint candidates. The acceleration and
repeat thresholds are local exploratory choices. Repeated sprints require observed
recovery intervals.

Rolling peaks require at least 80% observed segment exposure in a clock window.
Missing distance is not filled in. Counts of overlapping near-peak anchors are
not independent efforts or total minutes near peak. Camera-selected match load
does not establish conditioning capacity, fatigue or injury risk. The data has
no heart rate, RPE, training-load history or fitness tests. Methods and limitations
are saved alongside `player_external_load`, `movement_rotations`,
`movement_transitions`, `movement_phase_load` and `professional_method_sources`.

## Public research artifacts

`results/summary.json` and its PNG/SVG figures contain aggregate outcomes and report
hashes. The exporter omits raw records, cross-source identity links, individual test
predictions and model binaries. Reproduce the analyses locally with the commands
above. `--database` also archives these summaries and figures; `analysis_run_assets`
links each run to its versioned files. FM input is optional and must be supplied locally. Code uses the repository
MIT license; provider datasets and game data retain their own terms. Source URLs,
coverage and import status stay in the warehouse.

`.gitignore` and `.dockerignore` exclude the large database, downloads, logs and
local model files. The base Docker image includes prediction dependencies. Run
the research commands locally with `requirements-scouting.txt`, or build a research
environment with those extras and an explicitly mounted database. These results
are separate from the live dashboard export and its recommendation gate.

## Dashboard and release checks

`export-site` writes six files under `docs/data/`:

| File | Contents |
| --- | --- |
| `predictions.json` | Probabilities, explanations, model identities and partial errors |
| `preset_questions.json` | Recommended picks permitted by the release gate |
| `scores.json` | Daily scores and source errors |
| `operations.json` | Model coverage, validation, backtests, calibration and freshness |
| `release.json` | Release decision, blockers, thresholds, expiry and risk disclosure |
| `manifest.json` | Snapshot timestamp, run ID, file sizes and SHA-256 hashes |

Each file is replaced atomically, with the manifest written last. The dashboard verifies timestamps and hashes before showing the snapshot. Serve the site over localhost or HTTPS so browser SHA-256 verification is available:

```bash
python3 -m http.server 8000 --bind 127.0.0.1 --directory docs
```

Open `http://localhost:8000/` for the research overview. Match prediction and release checks are at `http://localhost:8000/predictions.html`. The five main pages separate matches, entities, models, and reference material; historical `research.html#...` links redirect with their query parameters.

`release.json` uses `SoccerAnalytics.ReleaseGovernance.v3`. Recommendations require an approved decision, `can_publish_recommendations: true`, no blockers, and an unexpired `expires_at`. Permission expires at the first fixture's kickoff or 24 hours after export, whichever comes first. Refresh the export to reassess remaining fixtures. The shortlist uses the operations high-confidence threshold, normally 60%.

Failed training leaves existing model files intact. Prediction can fall back to a previous bundle if the active file is missing or invalid; release checks block recommendations using a fallback, stale model, incomplete coverage or inadequate quality evidence. Model probabilities do not guarantee match outcomes.

## GitHub Actions

For football-data.org, set the repository Actions secret `FOOTBALL_DATA_API_KEY`. For key-free runs, set `SOCCER_ANALYTICS_DATA_SOURCE` and select supported codes with `SOCCER_ANALYTICS_COMPETITIONS` in the workflow environment. Configure GitHub Pages to deploy through Actions. `daily_update.yml` runs at 07:00 UTC, trains models and commits exported site data. `pages.yml` deploys `docs/` after a successful daily refresh, a push to `main`, or a manual run. The completion trigger handles daily commits made with `GITHUB_TOKEN`, which do not trigger a second push workflow.

## Verification

Model bundles use executable Python pickle and must come from trusted local
training. Before loading either version, the loader rejects symlinks, special
files, group/world-writable files or model directories, and file owners that
differ from the model-directory owner. Parent paths remain an operator trust
boundary; these checks do not make downloaded pickle safe. Training still writes
private files and aligns their ownership with the mounted model directory.

The [cross-field study](results/switching-play.json) joins 51 Euro2024 matches to
match-day coaches and opponent contexts. Run `python -m app.analytics.switching_play`
to rebuild the descriptive pass windows and store them in the warehouse.

`python -m app.analytics.switch_context` adds opposition shape, provider pressure
levels and detected teammate movement around SkillCorner transfers. Pre- and
post-release movement and possession changes remain separate.
[Context results and a pitch diagram](results/README.md#新增数据的分析发现)
include missing visibility and coaching evidence. Euro360 snapshots and continuous
A-League tracking remain separate cohorts.

Node.js is required for the dashboard runtime tests.

```bash
python scripts/check.py
python scripts/check.py --static-only --browser
```

Tests use controlled data and temporary model files. Install `requirements-experiments.txt` to include the optional model/search/fpstreams checks; otherwise those tests are explicitly skipped. Live commands require network access; the public sources need no API key. Training evaluates expanding time windows and a final chronological holdout against majority-class and Elo baselines before saving a model.

The optional SoccerData checks run in the collection environment with `python -m unittest tests.test_soccerdata_collect -v`. They cover season interpretation, browser reader configuration, ESPN home/away mapping and lossless dataframe column export. The base suite skips checks whose optional dependencies are absent.

## Maintenance interfaces

The default check command runs the existing unittest suite with a production-warehouse access guard, then checks Python syntax, all JavaScript sources, Shell scripts, local page resources, documentation links, and public-file identities. It never runs collection or production training commands. Fixture tests may fit bounded models. Optional dependencies gate only the checks that use them.

```bash
python scripts/check.py
python scripts/check.py tests.test_history tests.test_experiments
python scripts/check.py --static-only
python scripts/public_manifest.py --refresh
python scripts/public_manifest.py
```

Browser verification is explicit. Install Playwright and Chrome/Chromium separately, or point `PLAYWRIGHT_MODULE` and `PLAYWRIGHT_CHROMIUM_EXECUTABLE` at an existing local runtime. The script starts its own loopback server and isolated browser and closes both at exit. `FRONTEND_CASE` accepts comma-separated scenario names. An unknown name or an empty selection fails the check. `TOUCHLINE_BROWSER_OUTPUT` selects the evidence directory. No browser account or existing profile is used.

```bash
python scripts/check.py --static-only --browser
FRONTEND_CASE=failure-isolation-and-retry node tests/frontend_browser.cjs
```

Source collection handlers now live in `app/cli/collection.py`, scouting and tactical commands in `app/cli/scouting.py`, and match-model studies in `app/cli/model_research.py`. `app/cli/research.py` retains dispatch and legacy handler imports. Typed public projections live in `app/web/public_*.py`; `app/web/research.py` retains the export entry point and compatible function imports. Existing command names, output schemas, match classes, and tracking-reader imports remain available.

Raw sources, local experiment outputs, model weights, and database files retain their existing paths and identities. Bytecode caches can be regenerated; private runs and source receipts cannot be treated as disposable caches. The maintenance task does not rewrite historical results or publish new forecasts.
