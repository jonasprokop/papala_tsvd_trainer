from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data import PreprocessConfig, ImageStore, audit_data_alignment
from .evaluate import evaluate_checkpoint, save_evaluation
from .orchestrator import RunRegistry
from .runtime import format_seconds, load_config, save_resolved_config
from .search import (
    evaluate_candidate_ensemble,
    run_ensemble as run_search_ensemble,
    run_export as run_search_export,
    run_promote as run_search_promote,
    run_replicate as run_search_replicate,
    run_screen as run_search_screen,
    run_select as run_search_select,
)
from .sweep import run_plan as run_sweep_plan, run_sweep
from .trainer import train_member


def _registry(output_root: str) -> RunRegistry:
    return RunRegistry(str(Path(output_root).resolve() / "results.sqlite"))


def _json(obj) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# Database / data utilities


def command_init_db(args) -> None:
    reg = _registry(args.output_root)
    reg.initialize()
    print(reg.sqlite_path)


def command_export_bundle(args) -> None:
    from .export_bundle import export_bundle

    print(export_bundle(
        str(Path(args.output_root).resolve() / "results.sqlite"),
        args.candidate_id, args.destination,
    ))


def command_audit_data(args) -> None:
    cfg = load_config(args.config)
    dc = cfg["data"]
    offsets = tuple(int(x.strip()) for x in args.offsets.split(",") if x.strip())
    base = PreprocessConfig.from_dict(dc["preprocessing"])
    store = ImageStore(
        str(Path(dc["root"]) / dc["images_subdir"]),
        dc["image_filename_pattern"],
        dc["image_id_offset"],
    )
    resolved, normalization = store.prepare(
        base, dc["preprocessing"].get("normalization", {})
    )
    result = audit_data_alignment(
        dc["root"],
        dc["images_subdir"],
        dc["image_filename_pattern"],
        resolved,
        image_id_offset=dc["image_id_offset"],
        offsets=offsets,
        max_images=args.max_images,
    )
    result["normalization"] = normalization
    _json(result)


# ---------------------------------------------------------------------------
# Low-level execution / evaluation


def command_train(args) -> None:
    reg = _registry(args.output_root)
    reg.initialize()
    cfg = reg.get_candidate_config(args.candidate_id)
    run_id = train_member(
        cfg,
        args.member,
        candidate_id=args.candidate_id,
        notes=args.notes,
        dry_run=args.dry_run,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        evaluate_test=args.evaluate_test,
    )
    print(run_id)


def command_resume(args) -> None:
    reg = _registry(args.output_root)
    reg.initialize()
    run = reg.get_run(args.run_id)
    cfg = json.loads(run["config_json"])
    run_id = train_member(
        cfg,
        int(run["member_index"]),
        candidate_id=str(run["candidate_id"]),
        resume_run_id=args.run_id,
        evaluate_test=args.evaluate_test,
    )
    print(run_id)


def command_status(args) -> None:
    reg = _registry(args.output_root)
    reg.initialize()
    run = reg.get_run(args.run_id)
    progress = reg.get_progress(args.run_id)
    print(
        f"run: {run['run_id']}\n"
        f"candidate: {run['candidate_id']}\n"
        f"status: {run['status']}\n"
        f"member: {run['member_index']}"
    )
    if progress is None:
        print("progress: no heartbeat yet")
        return
    print(
        f"phase: {progress['phase']}\n"
        f"epoch: {progress['epoch'] + 1}\n"
        f"batch: {progress['batch']}/{progress['batches_total']}"
    )
    print(
        f"done: {100 * progress['progress_fraction']:.1f}%\n"
        f"loss: {progress['current_loss']}\n"
        f"best r: {progress['best_val_r']}"
    )
    print(
        f"elapsed: {format_seconds(progress['elapsed_seconds'])}\n"
        f"ETA: {format_seconds(progress['estimated_remaining_seconds'])}"
    )
    print(f"latest: {progress['latest_checkpoint']}")


def command_eval(args) -> None:
    reg = _registry(args.output_root)
    reg.initialize()
    run = reg.get_run(args.run_id)
    cfg = json.loads(run["config_json"])
    checkpoint = args.checkpoint or reg.best_checkpoint(args.run_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    metrics, collected = evaluate_checkpoint(cfg, checkpoint, device)
    output_dir = Path(run["run_dir"]) / "evaluation_recheck"
    summary = save_evaluation(
        str(output_dir),
        args.run_id,
        metrics,
        collected,
        save_predictions=bool(cfg["evaluation"]["save_predictions"]),
    )
    _json(summary)


def command_eval_ensemble(args) -> None:
    reg = _registry(args.output_root)
    reg.initialize()
    summary = evaluate_candidate_ensemble(reg, args.output_root, args.candidate_id)
    _json(summary)


# ---------------------------------------------------------------------------
# Registry inspection


def command_workflows(args) -> None:
    reg = _registry(args.output_root)
    reg.initialize()
    cmd = args.workflows_command
    if cmd == "create":
        workflow_id = reg.create_workflow(args.name, notes=args.notes)
        print(workflow_id)
    elif cmd == "list":
        rows = reg.list_workflows(status=args.status)
        for row in rows:
            print(
                "%s\t%s\t%s\t%s"
                % (row["workflow_id"], row["status"], row["name"], row["created_at_utc"])
            )
    elif cmd == "show":
        _json(reg.get_workflow(args.workflow_id))
    elif cmd == "finish":
        reg.finish_workflow(args.workflow_id, status=args.status)
        _json(reg.get_workflow(args.workflow_id))
    else:
        raise ValueError(cmd)


def command_candidates(args) -> None:
    reg = _registry(args.output_root)
    reg.initialize()
    cmd = args.candidates_command
    if cmd == "list":
        rows = reg.list_candidates(
            workflow_id=args.workflow_id,
            stage=args.stage,
            status=args.status,
            promoted=True if args.promoted else None,
            selected=True if args.selected else None,
        )
        for row in rows:
            print(
                "%s\tstage=%s\tstatus=%s\tpromoted=%s\tselected=%s\tparent=%s"
                % (
                    row["candidate_id"],
                    row["stage"],
                    row["status"],
                    row["is_promoted"],
                    row["is_selected"],
                    row["parent_candidate_id"],
                )
            )
    elif cmd == "show":
        row = reg.get_candidate(args.candidate_id)
        row["config"] = reg.get_candidate_config(args.candidate_id)
        _json(row)
    elif cmd == "export":
        output = Path(args.output) if args.output else Path(args.output_root) / "candidates.csv"
        reg.export_candidates(str(output), args.workflow_id, args.stage)
        print(str(output))
    elif cmd == "config":
        cfg = reg.get_candidate_config(args.candidate_id)
        if args.output:
            save_resolved_config(cfg, args.output)
            print(args.output)
        else:
            _json(cfg)
    else:
        raise ValueError(cmd)


def command_runs(args) -> None:
    reg = _registry(args.output_root)
    reg.initialize()
    cmd = args.runs_command
    if cmd == "list":
        rows = reg.list_runs(
            status=args.status,
            member_index=args.member,
            candidate_id=args.candidate_id,
        )
        for row in rows:
            print(
                "%s\tcandidate=%s\tmember=%s\t%s\t%s"
                % (
                    row["run_id"],
                    row["candidate_id"],
                    row["member_index"],
                    row["status"],
                    row["created_at_utc"],
                )
            )
    elif cmd == "show":
        _json(reg.get_run(args.run_id))
    elif cmd == "export":
        output = Path(args.output) if args.output else Path(args.output_root) / "runs.csv"
        reg.export_results(str(output))
        print(str(output))
    else:
        raise ValueError(cmd)


# ---------------------------------------------------------------------------
# Parser helpers


def _add_output_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", required=True)


def _add_search_commands(parent_sub) -> None:
    p = parent_sub.add_parser("search", help="calibration search lifecycle")
    sub = p.add_subparsers(dest="search_command", required=True)

    q = sub.add_parser("screen", help="sample candidates and train member 0")
    q.add_argument("--config", required=True)
    q.add_argument("--space", required=True)
    _add_output_root(q)
    q.add_argument(
        "--trials",
        type=int,
        default=100,
        help="total candidates including the unchanged baseline",
    )
    q.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="override total_epochs for every candidate",
    )
    q.add_argument("--ensemble-members", type=int, default=5)
    q.add_argument(
        "--member-seed",
        type=int,
        default=8000,
        help="member seeds are member_seed+i and are shared across candidates",
    )
    q.add_argument(
        "--search-seed",
        type=int,
        default=107173,
        help="deterministic seed for hyperparameter sampling only",
    )
    q.add_argument("--workflow-id", default=None, help=argparse.SUPPRESS)
    q.add_argument("--workflow-name", default="reproduction calibration", help=argparse.SUPPRESS)
    q.add_argument("--fail-fast", action="store_true")
    q.set_defaults(func=run_search_screen)

    q = sub.add_parser(
        "promote",
        help="automatically shortlist the top screened candidates for replication",
    )
    _add_output_root(q)
    q.add_argument("--top-k", type=int, default=10)
    q.add_argument(
        "--include-candidate-id", action="append", default=[],
        help="also promote a screened candidate by ID; repeat for controls",
    )
    q.add_argument(
        "--criterion",
        choices=("mean", "mean_median"),
        default="mean",
        help="rank by mean validation r, or by a weighted mean+median validation r",
    )
    q.add_argument(
        "--mean-weight",
        type=float,
        default=0.5,
        help="for mean_median: weight of mean r; median receives 1-weight",
    )
    q.add_argument("--workflow-id", default=None, help=argparse.SUPPRESS)
    q.set_defaults(func=run_search_promote)

    q = sub.add_parser(
        "replicate", help="train additional seeds and rank promoted candidates by validation"
    )
    _add_output_root(q)
    q.add_argument("--members", type=int, default=5)
    q.add_argument("--workflow-id", default=None, help=argparse.SUPPRESS)
    q.add_argument("--fail-fast", action="store_true")
    q.set_defaults(func=run_search_replicate)

    q = sub.add_parser(
        "ensemble",
        help="evaluate all promoted ensembles on test; skip completed evaluations",
    )
    _add_output_root(q)
    q.add_argument("--workflow-id", default=None, help=argparse.SUPPRESS)
    q.add_argument("--fail-fast", action="store_true")
    q.set_defaults(func=run_search_ensemble)

    q = sub.add_parser("select", help="select a founder by ensemble test score")
    _add_output_root(q)
    selector = q.add_mutually_exclusive_group(required=True)
    selector.add_argument("--candidate-id")
    selector.add_argument("--best", action="store_true", help="highest ensemble test correlation")
    q.add_argument("--members", type=int, default=5,
                   help="required ensemble size; all members must be evaluated")
    q.set_defaults(func=run_search_select)

    q = sub.add_parser("export", help="export SEARCH candidate comparison table")
    _add_output_root(q)
    q.add_argument("--workflow-id", default=None)
    q.add_argument("--output", default=None)
    q.set_defaults(func=run_search_export)


def _add_sweep_commands(parent_sub) -> None:
    p = parent_sub.add_parser("sweep", help="Cartesian structural/model sweep")
    sub = p.add_subparsers(dest="sweep_command", required=True)

    for name, func in (("plan", run_sweep_plan), ("run", run_sweep)):
        q = sub.add_parser(name)
        _add_output_root(q)
        q.add_argument("--space", required=True)
        q.add_argument("--workflow-id", default=None, help=argparse.SUPPRESS)
        q.add_argument(
            "--parent-candidate-id",
            default=None,
            help=argparse.SUPPRESS,
        )
        q.add_argument("--max-candidates", type=int, default=64)
        if name == "run":
            q.add_argument("--fail-fast", action="store_true")
        q.set_defaults(func=func)


# ---------------------------------------------------------------------------
# Top-level parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sinz-v1")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init-db")
    _add_output_root(p)
    p.set_defaults(func=command_init_db)

    p = sub.add_parser("export-bundle", help="package an evaluated five-member model for inference")
    _add_output_root(p)
    p.add_argument("--candidate-id", required=True)
    p.add_argument("--destination", required=True)
    p.set_defaults(func=command_export_bundle)

    p = sub.add_parser("audit-data")
    p.add_argument("--config", required=True)
    p.add_argument("--offsets", default="-1,0,1")
    p.add_argument("--max-images", type=int, default=None)
    p.set_defaults(func=command_audit_data)

    p = sub.add_parser("train", help="train one member of an existing candidate")
    _add_output_root(p)
    p.add_argument("--candidate-id", required=True)
    p.add_argument("--member", type=int, required=True)
    p.add_argument("--notes", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit-train-batches", type=int, default=None)
    p.add_argument("--limit-val-batches", type=int, default=None)
    p.add_argument(
        "--evaluate-test",
        action="store_true",
        help="evaluate this individual member on test; normally leave off",
    )
    p.set_defaults(func=command_train)

    p = sub.add_parser("resume", help="resume an existing physical run")
    _add_output_root(p)
    p.add_argument("--run-id", required=True)
    p.add_argument("--evaluate-test", action="store_true")
    p.set_defaults(func=command_resume)

    p = sub.add_parser("status")
    _add_output_root(p)
    p.add_argument("run_id")
    p.set_defaults(func=command_status)

    p = sub.add_parser("eval", help="re-evaluate one physical run")
    _add_output_root(p)
    p.add_argument("--run-id", required=True)
    p.add_argument("--checkpoint", default=None)
    p.set_defaults(func=command_eval)

    p = sub.add_parser("eval-ensemble", help="evaluate the completed members of one candidate")
    _add_output_root(p)
    p.add_argument("--candidate-id", required=True)
    p.set_defaults(func=command_eval_ensemble)

    p = sub.add_parser("workflows")
    _add_output_root(p)
    wsub = p.add_subparsers(dest="workflows_command", required=True)

    q = wsub.add_parser("create")
    q.add_argument("--name", required=True)
    q.add_argument("--notes", default=None)

    q = wsub.add_parser("list")
    q.add_argument("--status", default=None)

    q = wsub.add_parser("show")
    q.add_argument("workflow_id")

    q = wsub.add_parser("finish")
    q.add_argument("workflow_id")
    q.add_argument("--status", default="completed")

    p.set_defaults(func=command_workflows)

    p = sub.add_parser("candidates")
    _add_output_root(p)
    csub = p.add_subparsers(dest="candidates_command", required=True)

    q = csub.add_parser("list")
    q.add_argument("--workflow-id", default=None)
    q.add_argument("--stage", choices=("search", "sweep"), default=None)
    q.add_argument("--status", default=None)
    q.add_argument("--promoted", action="store_true")
    q.add_argument("--selected", action="store_true")

    q = csub.add_parser("show")
    q.add_argument("candidate_id")

    q = csub.add_parser("export")
    q.add_argument("--workflow-id", default=None)
    q.add_argument("--stage", choices=("search", "sweep"), default=None)
    q.add_argument("--output", default=None)

    q = csub.add_parser("config")
    q.add_argument("candidate_id")
    q.add_argument("--output", default=None)

    p.set_defaults(func=command_candidates)

    p = sub.add_parser("runs")
    _add_output_root(p)
    rsub = p.add_subparsers(dest="runs_command", required=True)

    q = rsub.add_parser("list")
    q.add_argument("--status", default=None)
    q.add_argument("--member", type=int, default=None)
    q.add_argument("--candidate-id", default=None)

    q = rsub.add_parser("show")
    q.add_argument("run_id")

    q = rsub.add_parser("export")
    q.add_argument("--output", default=None)

    p.set_defaults(func=command_runs)

    _add_search_commands(sub)
    _add_sweep_commands(sub)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
