"""Independent research CLI; data tools and help need no prediction stack."""

import os

from app.cli.arguments import build_research_parser
from app.cli.research import run_research_command


def main(argv=None):
    args = build_research_parser().parse_args(argv)
    if args.source:
        os.environ["SOCCER_ANALYTICS_DATA_SOURCE"] = args.source
    trainer_type = None
    if args.command == "experiment":
        from app.ml.training import ModelTrainer

        trainer_type = ModelTrainer
    result = run_research_command(args, trainer_type=trainer_type)
    if result is None:
        raise AssertionError("未知研究命令")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
