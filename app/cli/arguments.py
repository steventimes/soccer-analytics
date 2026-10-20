"""CLI options shared by the production and research entry points."""

import argparse
import json
from pathlib import Path

from app.config import parse_training_seasons


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须大于 0")
    return parsed


def _add_competition_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--competitions",
        help="逗号分隔的联赛代码，例如 PL,PD,SA。",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "football-data", "football-csv", "openfootball"),
        help="数据源；auto 在未设置 API key 时使用公开 CSV。",
    )


def _add_assessment_commands(subparsers):
    assessment_parser = subparsers.add_parser(
        "assess", help="建立球员和球队评估、风格模型，并离线比较多种 ML 方法。"
    )
    assessment_parser.set_defaults(source=None)
    assessment_parser.add_argument(
        "--database", type=Path, default=Path("data/football.sqlite3")
    )
    assessment_parser.add_argument(
        "--output-dir", type=Path, default=Path(".tmp/assessments")
    )
    assessment_parser.add_argument(
        "--models",
        default="ridge,bayesian_ridge,elastic_net,sgd,svr,random_forest,extra_trees,hist_gradient_boosting,xgboost,mlp",
    )
    assessment_parser.add_argument("--train-cap", type=_positive_int, default=5000)

    insight_parser = subparsers.add_parser(
        "scout-insights", help="建立球员习惯卡、年轻球员观察名单和对手备战档案。"
    )
    insight_parser.set_defaults(source=None)
    insight_parser.add_argument(
        "--database", type=Path, default=Path("data/football.sqlite3")
    )
    insight_parser.add_argument(
        "--output-dir", type=Path, default=Path(".tmp/scouting-intelligence")
    )
    insight_parser.add_argument(
        "--assessment-dir", type=Path, default=Path(".tmp/assessments")
    )
    insight_parser.add_argument(
        "--player-as-of",
        choices=("2017-10-31", "2017-12-31", "2018-02-28", "2018-05-31"),
        default="2018-05-31",
    )
    insight_parser.add_argument(
        "--backend", choices=("python", "fpstreams", "compare"), default="python"
    )

    for command, folder, help_text in [
        (
            "tactics",
            "tactical-relations",
            "研究阵型、进攻方向、控球和对手，并比较机会创造模型。",
        ),
        ("movement-lab", "movement-lab", "联动轮转候选、攻守转换与观测体能负荷。"),
    ]:
        analysis_parser = subparsers.add_parser(command, help=help_text)
        analysis_parser.set_defaults(source=None)
        analysis_parser.add_argument(
            "--database", type=Path, default=Path("data/football.sqlite3")
        )
        analysis_parser.add_argument(
            "--output-dir", type=Path, default=Path(".tmp") / folder
        )
        if command == "tactics":
            analysis_parser.add_argument(
                "--train-cap", type=_positive_int, default=15000
            )


def _add_prediction_commands(subparsers):
    train_parser = subparsers.add_parser("train", help="训练并保存所选联赛模型。")
    _add_competition_argument(train_parser)
    train_parser.add_argument(
        "--seasons",
        type=parse_training_seasons,
        help="逗号分隔的赛季起始年，例如 2022,2023,2024。",
    )
    train_parser.add_argument(
        "--model",
        choices=("xgboost", "sgd", "random_forest", "gaussian_nb",
                 "logistic_l1", "logistic_l2", "logistic_elasticnet", "elo_ordered_logit"),
        default="xgboost",
    )
    train_parser.add_argument(
        "--params", type=json.loads, default={}, help="模型超参数 JSON。"
    )

    predict_parser = subparsers.add_parser("predict", help="预测未来赛程。")
    _add_competition_argument(predict_parser)
    predict_parser.add_argument("--days", type=_positive_int)

    export_parser = subparsers.add_parser(
        "export-site",
        help="预测并原子导出静态仪表盘数据。",
    )
    _add_competition_argument(export_parser)
    export_parser.add_argument("--days", type=_positive_int, default=None)

    full_parser = subparsers.add_parser(
        "all",
        help="训练所选联赛并导出仪表盘。",
    )
    _add_competition_argument(full_parser)
    full_parser.add_argument("--seasons", type=parse_training_seasons)
    full_parser.add_argument("--days", type=_positive_int)
    full_parser.add_argument(
        "--model",
        choices=("xgboost", "sgd", "random_forest", "gaussian_nb",
                 "logistic_l1", "logistic_l2", "logistic_elasticnet", "elo_ordered_logit"),
        default="xgboost",
    )
    full_parser.add_argument("--params", type=json.loads, default={})


def _add_experiment_commands(subparsers):
    for command in ("analyze", "experiment", "calibrate"):
        experiment_parser = subparsers.add_parser(
            command,
            help=(
                "球队统计分析。"
                if command == "analyze"
                else "按时间评估模型及优化方法，不覆盖生产模型。"
                if command == "experiment"
                else "按独立时间窗口拟合并评估概率校准，不覆盖生产模型。"
            ),
        )
        _add_competition_argument(experiment_parser)
        experiment_parser.add_argument("--seasons", type=parse_training_seasons)
        experiment_parser.add_argument(
            "--output", type=Path, help="JSON 输出路径；默认输出到终端。"
        )
        if command == "analyze":
            experiment_parser.add_argument(
                "--backend",
                choices=("python", "fpstreams", "compare"),
                default="python",
            )
        elif command == "calibrate":
            experiment_parser.add_argument("--model", default="xgboost")
            experiment_parser.add_argument("--params", type=json.loads, default={})
            experiment_parser.add_argument("--calibrators", default="identity,temperature,dirichlet_l2",
                                           help="逗号分隔：identity,temperature,dirichlet_l2,dirichlet_odir,draw_constant,draw_logistic,draw_beta；ODIR 和平局方法需显式选择。")
            experiment_parser.add_argument("--calibration-fraction", type=float, default=0.2)
            experiment_parser.add_argument("--folds", type=_positive_int, default=3)
            experiment_parser.add_argument("--seed", type=int, default=42)
            experiment_parser.add_argument(
                "--snapshots", type=Path,
                help="可选本地校验快照目录，布局为 <competition>/<season>.json；需明确 --source。",
            )
        else:
            experiment_parser.add_argument(
                "--models", default="xgboost,sgd,random_forest,gaussian_nb"
            )
            experiment_parser.add_argument(
                "--optimizers", default="bayesian,genetic",
                help="逗号分隔的优化方法：bayesian、genetic、optuna。",
            )
            experiment_parser.add_argument("--trials", type=_positive_int, default=12)
            experiment_parser.add_argument("--seed", type=int, default=42)

    ratings = subparsers.add_parser("rating-baselines", help="比较 Elo 有序/多项逻辑回归与完整特征对照。")
    _add_competition_argument(ratings)
    ratings.add_argument("--seasons", type=parse_training_seasons)
    ratings.add_argument("--output", type=Path)
    ratings.add_argument("--snapshots", type=Path, help="已校验赛季快照根目录；需明确 --source。")
    ratings.add_argument("--ridge", type=float, default=.001)
    ratings.add_argument("--folds", type=_positive_int, default=3)
    ratings.add_argument("--seed", type=int, default=42)
    ratings.add_argument("--ratings", default="basic", help="basic,goal_weighted,pi 的逗号列表；默认保持原有基本 Elo 对照。")
    ratings.add_argument("--warmup-matches", type=int, default=0, help="每队至少完成的预热场数，所有配置使用相同入选条件。")

    explanations = subparsers.add_parser("explain-model", help="固定时间留出中的全局 TreeSHAP、特征依赖和显式交互，不覆盖生产模型。")
    _add_competition_argument(explanations)
    explanations.add_argument("--seasons", type=parse_training_seasons)
    explanations.add_argument("--snapshots", type=Path, help="已校验快照根目录；需明确 --source。")
    explanations.add_argument("--output-dir", type=Path, default=Path(".tmp/global-explanations"))
    explanations.add_argument("--class", dest="class_label", choices=("A", "D", "H"), default="H")
    explanations.add_argument("--against", choices=("A", "D", "H"))
    explanations.add_argument("--max-rows", type=_positive_int, default=120)
    explanations.add_argument("--seed", type=int, default=42)

    features = subparsers.add_parser("feature-experiment", help="联合搜索近期窗口、Elo 与模型参数，不覆盖生产模型。")
    _add_competition_argument(features)
    features.add_argument("--seasons", type=parse_training_seasons)
    features.add_argument("--output", type=Path)
    features.add_argument("--snapshots", type=Path, help="已校验赛季快照根目录；需明确 --source。")
    features.add_argument("--models", default="xgboost,logistic_l2")
    features.add_argument("--optimizer", choices=("bayesian", "genetic", "optuna"), default="optuna")
    features.add_argument("--trials", type=_positive_int, default=12)
    features.add_argument("--folds", type=_positive_int, default=3)
    features.add_argument("--seed", type=int, default=42)

    score = subparsers.add_parser("score-models", help="离线或联网比较 Poisson 与 Dixon–Coles 比分模型。")
    _add_competition_argument(score)
    score.add_argument("--seasons", type=parse_training_seasons)
    score.add_argument("--output", type=Path)
    score.add_argument("--snapshots", type=Path, help="已校验赛季快照根目录；需明确 --source。")
    score.add_argument("--models", default="poisson,poisson_decay,dixon_coles",
                       help="逗号分隔的比分模型；可显式加入共用进球率 poisson_constant 或共享分量 bivariate_poisson。")
    score.add_argument("--half-life-days", type=float, default=365.)
    score.add_argument("--ridge", type=float, default=.01)
    score.add_argument("--folds", type=_positive_int, default=3)
    score.add_argument("--max-iter", type=_positive_int, default=1000)

    diagnostics = subparsers.add_parser("calibration-diagnostics", help="诊断本地固定赛果概率的边际、联合校准和分箱敏感性。")
    diagnostics.set_defaults(source=None)
    diagnostics.add_argument("--predictions", type=Path, required=True, help="数值NPZ：targets、probabilities和model_names。")
    diagnostics.add_argument("--bins", default="5,10,20")
    diagnostics.add_argument("--minimum-bin-count", type=_positive_int, default=10)
    diagnostics.add_argument("--output", type=Path)

    comparison = subparsers.add_parser("compare-outcomes", help="比较本地固定赛果概率的配对损失与时间相关敏感性。")
    comparison.set_defaults(source=None)
    comparison.add_argument("--predictions", type=Path, required=True, help="同一比赛队列的数值 NPZ；完整字段见复现说明。")
    comparison.add_argument("--reference", required=True, help="NPZ 内的固定参照模型名。")
    comparison.add_argument("--output", type=Path)
    comparison.add_argument("--replicates", type=_positive_int, default=2000)
    comparison.add_argument("--seed", type=int, default=20261017)
    comparison.add_argument("--block-lengths", default="1,3,7", help="按观测比赛日计的平均块长度。")
    comparison.add_argument("--hac-lags", default="0,1,3,7", help="Bartlett 方差的滞后长度。")


def _add_history_commands(subparsers):
    for command in ("collect-history", "compare-history"):
        history_parser = subparsers.add_parser(
            command,
            help=(
                "缓存公开历史赛季。"
                if command == "collect-history"
                else "离线进行多赛季和跨联赛对比。"
            ),
        )
        _add_competition_argument(history_parser)
        history_parser.add_argument("--sources", default="football-csv,openfootball")
        history_parser.add_argument(
            "--cache-dir", type=Path, default=Path(".tmp/history")
        )
        history_parser.add_argument("--output", type=Path)
        if command == "collect-history":
            history_parser.add_argument("--seasons", type=parse_training_seasons)
            history_parser.add_argument("--refresh", action="store_true")
            history_parser.add_argument("--max-age-hours", type=float, help="所选赛季的显式缓存寿命；默认固定快照。")
            history_parser.add_argument("--active-max-age-hours", type=float, help="当前及未来赛季的缓存寿命，覆盖全局寿命。")
        else:
            history_parser.add_argument("--test-seasons", type=parse_training_seasons)
            history_parser.add_argument("--windows", default="1,3,all")
            history_parser.add_argument(
                "--models", default="xgboost,sgd,random_forest,gaussian_nb"
            )
            history_parser.add_argument("--seed", type=int, default=42)
            history_parser.add_argument("--no-pool", action="store_true")


def _add_scouting_commands(subparsers):
    for command in (
        "collect-open-data",
        "scout-events",
        "scout-tracking",
        "scout-bodypose",
        "scout-figures",
    ):
        scout_parser = subparsers.add_parser(
            command,
            help={
                "collect-open-data": "采集所有公开档案、事件和追踪数据。",
                "scout-events": "分析事件、传球网络、360和射门模型。",
                "scout-tracking": "分析连续追踪和无球跑动。",
                "scout-bodypose": "检查完整3D姿态覆盖与不确定度。",
                "scout-figures": "生成来源明确的球场分析图。",
            }[command],
        )
        scout_parser.set_defaults(source=None)
        scout_parser.add_argument(
            "--data-dir", type=Path, default=Path(".tmp/open-data")
        )
        if command == "collect-open-data":
            scout_parser.add_argument(
                "--datasets",
                default="football-csv,openfootball,statsbomb,metrica,skillcorner,bodypose",
            )
            scout_parser.add_argument("--refresh", action="store_true")
            scout_parser.add_argument("--max-age-hours", type=float, help="仅 football-csv ZIP 的显式缓存寿命。")
            scout_parser.add_argument("--active-max-age-hours", type=float, help="仅当前及未来 football-csv 赛季的缓存寿命。")
            scout_parser.add_argument("--output", type=Path)
        else:
            folder = {
                "scout-events": "statsbomb",
                "scout-tracking": "movement",
                "scout-bodypose": "movement",
                "scout-figures": "figures",
            }[command]
            scout_parser.add_argument(
                "--output-dir", type=Path, default=Path(".tmp/scouting") / folder
            )
            if command == "scout-figures":
                scout_parser.add_argument(
                    "--network-window",
                    nargs=2,
                    type=int,
                    metavar=("START", "END"),
                    help="只对传球网络选择第一半场分钟范围，例如10 40。",
                )
                scout_parser.add_argument(
                    "--scouting-dir", type=Path, default=Path(".tmp/scouting")
                )
                scout_parser.add_argument("--match-id", type=_positive_int)
            if command == "scout-events":
                scout_parser.add_argument("--maximum-matches", type=_positive_int)
                scout_parser.add_argument("--competition-id", type=_positive_int)
                scout_parser.add_argument("--season-id", type=_positive_int)


def _parser(description):
    parser = argparse.ArgumentParser(description=description)
    return parser, parser.add_subparsers(dest="command", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser, subparsers = _parser(
        "Soccer Analytics：足球数据、球探、战术、球员发展与预测分析。"
    )
    _add_assessment_commands(subparsers)
    _add_prediction_commands(subparsers)
    _add_experiment_commands(subparsers)
    _add_history_commands(subparsers)
    _add_scouting_commands(subparsers)
    return parser


def build_research_parser() -> argparse.ArgumentParser:
    parser, subparsers = _parser("足球研究：数据、球员与球队评估、战术和表示学习。")
    _add_assessment_commands(subparsers)
    _add_experiment_commands(subparsers)
    _add_history_commands(subparsers)
    _add_scouting_commands(subparsers)
    lab = subparsers.add_parser(
        "representation-lab", help="离线比较跨联赛迁移、多任务和自监督表示学习。"
    )
    lab.set_defaults(source=None)
    lab.add_argument("--database", type=Path, default=Path("data/football.sqlite3"))
    lab.add_argument("--output-dir", type=Path, default=Path(".tmp/representation-lab"))
    lab.add_argument(
        "--target-league",
        choices=("England", "France", "Germany", "Italy", "Spain"),
        default="England",
    )
    lab.add_argument("--train-cap", type=_positive_int, default=12000)
    lab.add_argument("--epochs", type=_positive_int, default=25)
    lab.add_argument("--seeds", default="11,29,47")
    for command, help_text in (
        ("passing-lab", "按出球时刻分析传球选项、供应商 EPV 和压迫链。"),
        ("value-lab", "用时间留出比赛验证 xT 和得分、失球事件价值。"),
        ("space-lab", "保留追踪帧身份，比较空间控制和时钟同步。"),
        ("buildup-lab", "区分球员风格与表现，验证组织贡献的跨赛季稳定性。"),
    ):
        experiment = subparsers.add_parser(command, help=help_text)
        experiment.set_defaults(source=None)
        experiment.add_argument("--database", type=Path, default=Path("data/football.sqlite3"))
        experiment.add_argument("--output-dir", type=Path, default=Path(".tmp") / command)
        if command == "passing-lab":
            experiment.add_argument("--data-dir", type=Path, default=Path(".tmp/open-data/skillcorner/data/matches"))
        elif command == "space-lab":
            experiment.add_argument("--data-dir", type=Path, default=Path(".tmp/open-data/metrica/data"))
            experiment.add_argument("--samples", type=_positive_int, default=240)
        elif command == "value-lab":
            experiment.add_argument("--target-league", choices=("England", "France", "Germany", "Italy", "Spain"), default="England")
            experiment.add_argument("--train-cap", type=_positive_int, default=12000)
            experiment.add_argument("--epochs", type=_positive_int, default=12)
            experiment.add_argument("--seeds", default="11,29,47")
            experiment.add_argument("--horizon", type=_positive_int, default=10)
        else:
            experiment.add_argument("--minimum-minutes", type=_positive_int, default=900)
    export = subparsers.add_parser(
        "export-research", help="校验并导出公开研究摘要到本地研究页面。"
    )
    export.set_defaults(source=None)
    export.add_argument("--results-dir", type=Path, default=Path("results"))
    export.add_argument("--output-dir", type=Path, default=Path("docs/data/research"))
    export.add_argument(
        "--experiment-report", type=Path, help="可选完整实验报告路径，仅导出聚合指标。"
    )
    return parser
