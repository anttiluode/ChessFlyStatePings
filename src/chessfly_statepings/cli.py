"""Command-line interface for ChessFlyStatePings."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch

from .arena import play_paired_arena
from .assets import ensure_artifacts
from .compare import compare_positions
from .encoding import canonical_fen, encode_fen
from .graph import ChessFlyGraph
from .model import ChessFlyBaseline
from .orthogonal import evaluate_orthogonal_positions
from .policy import ChessFlyPolicy
from .query_memory import evaluate_query_memory
from .receipts import build_receipt, write_receipt
from .specificity import evaluate_directional_specificity
from .state_ping import StatePingModel, trajectory_summary
from .weights import ChessFlyWeights

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
DEFAULT_RHOS = (0.25, 0.50, 0.75, 0.90)
DEFAULT_KAPPAS = (-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20)


def default_sweep() -> tuple[tuple[float, float], ...]:
    return tuple((rho, kappa) for rho in DEFAULT_RHOS for kappa in DEFAULT_KAPPAS)


def _common(parser: argparse.ArgumentParser, *, output: bool = True) -> None:
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--space-revision", default="main")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--device", default="auto")
    if output:
        parser.add_argument("--output", default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chessfly-statepings", description="History-bearing dynamics experiments on published ChessFly")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("assets", help="download and verify external ChessFly artifacts"); _common(p, output=False); p.add_argument("--force", action="store_true")
    p = sub.add_parser("probe", help="observe baseline fast/slow trajectory without intervention"); _common(p); p.add_argument("--fen", required=True); p.add_argument("--rho", type=float, default=0.75)
    p = sub.add_parser("compare", help="paired baseline/StatePing inference on FENs"); _common(p); p.add_argument("--positions", required=True); p.add_argument("--rho", type=float, default=0.75); p.add_argument("--kappa", type=float, default=0.10)
    p = sub.add_parser("sweep", help="run the declared rho/kappa grid over paired positions"); _common(p); p.add_argument("--positions", required=True)
    p = sub.add_parser("arena", help="paired-color raw-policy headless games"); _common(p); p.add_argument("--games", type=int, default=20); p.add_argument("--rho", type=float, default=0.75); p.add_argument("--kappa", type=float, default=0.10); p.add_argument("--seed", type=int, default=0); p.add_argument("--max-plies", type=int, default=300); p.add_argument("--openings", default=None)
    p = sub.add_parser("orthogonal", help="test readout history after removing the present-state direction"); _common(p); p.add_argument("--positions", required=True); p.add_argument("--rho", type=float, default=0.75); p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("specificity", help="rank the real orthogonal history direction against many matched controls"); _common(p); p.add_argument("--positions", required=True); p.add_argument("--rho", type=float, default=0.75); p.add_argument("--seed", type=int, default=0); p.add_argument("--controls", type=int, default=32); p.add_argument("--magnitudes", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    p = sub.add_parser("query-memory", help="ask whether the final state-bearing ping retrieves its own recorded settling history"); _common(p); p.add_argument("--positions", required=True); p.add_argument("--rho", type=float, default=0.75); p.add_argument("--seed", type=int, default=0)
    return parser


def _load_models(manifest: Any, device: str):
    graph = ChessFlyGraph.from_files(manifest.paths.connectome, manifest.paths.neurons)
    weights = ChessFlyWeights.from_file(manifest.paths.weights, device="cpu")
    return graph, weights, ChessFlyBaseline(graph, weights, device=device), StatePingModel(graph, weights, device=device)


def _model_metadata(weights: ChessFlyWeights) -> dict[str, Any]:
    return {"step": weights.step, "steps": weights.steps, "alpha": weights.alpha, "hidden": weights.hidden, "encoder_inputs": weights.encoder_inputs, "readout_neurons": weights.readout_neurons}


def _read_fens(path: str | Path) -> list[str]:
    rows=[]
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line=line.strip()
        if line and not line.startswith("#"): rows.append(line)
    if not rows: raise ValueError("position/opening file contains no FENs")
    return rows


def _output_path(command: str, requested: str | None) -> Path:
    if requested: return Path(requested)
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("results") / f"{command}-{stamp}.json"


def _acquire(args, loader):
    return loader(args.cache_dir, space_revision=args.space_revision, model_revision=args.model_revision, force=getattr(args,"force",False))


def _retrieval_payload(metrics: Any) -> dict[str, float]:
    return {
        "accuracy": float(metrics.accuracy),
        "mean_reciprocal_rank": float(metrics.mean_reciprocal_rank),
        "mean_correct_margin": float(metrics.mean_correct_margin),
    }


def main(argv: list[str] | None = None, *, artifact_loader: Callable[..., Any] = ensure_artifacts, model_loader: Callable[..., Any] = _load_models) -> int:
    args=build_parser().parse_args(argv)
    manifest=_acquire(args, artifact_loader)
    if args.command == "assets":
        print(json.dumps(manifest.to_dict(), indent=2, default=str)); return 0
    graph, weights, baseline, stateping = model_loader(manifest, args.device)
    if args.command == "probe":
        fen,_=canonical_fen(args.fen); x=torch.tensor(encode_fen(fen)); result=baseline.forward(x,include_activity=True)
        payload={"instability":result.instability,"trajectory":trajectory_summary(result.activity,rho=args.rho,readout_indices=baseline.readout_index)}
        receipt=build_receipt(command="probe",arguments=vars(args),device=str(baseline.device),artifact_manifest=manifest,model_metadata=_model_metadata(weights),inputs=[args.fen],results=payload,instability_count=int(result.instability is not None))
    elif args.command in {"compare","sweep"}:
        fens=_read_fens(args.positions); baseline_policy=ChessFlyPolicy(baseline)
        if args.command == "compare":
            state_policy=ChessFlyPolicy(stateping,forward_kwargs={"rho":args.rho,"kappa":args.kappa}); result=compare_positions(fens,baseline_policy,state_policy)
            payload={"rows":[asdict(r) for r in result.rows],"aggregate":dict(result.aggregate)}
            instability_count=int(result.aggregate.get("instability_count", 0))
        else:
            runs=[]
            instability_count=0
            for rho,kappa in default_sweep():
                result=compare_positions(fens,baseline_policy,ChessFlyPolicy(stateping,forward_kwargs={"rho":rho,"kappa":kappa}))
                runs.append({"rho":rho,"kappa":kappa,"aggregate":dict(result.aggregate)})
                instability_count += int(result.aggregate.get("instability_count", 0))
            payload={"runs":runs,"settings":len(runs)}
        receipt=build_receipt(command=args.command,arguments=vars(args),device=str(baseline.device),artifact_manifest=manifest,model_metadata=_model_metadata(weights),inputs=fens,results=payload,instability_count=instability_count)
    elif args.command == "orthogonal":
        fens=_read_fens(args.positions)
        result=evaluate_orthogonal_positions(fens, baseline, rho=args.rho, seed=args.seed)
        payload={
            "rho":result.rho,
            "seed":result.seed,
            "settings":len(result.runs),
            "runs":[
                {
                    "lambda":run.lambda_value,
                    "aggregate":dict(run.aggregate),
                    "rows":[asdict(row) for row in run.rows],
                }
                for run in result.runs
            ],
        }
        receipt=build_receipt(command="orthogonal",arguments=vars(args),device=str(baseline.device),artifact_manifest=manifest,model_metadata=_model_metadata(weights),inputs=fens,results=payload,instability_count=0)
    elif args.command == "specificity":
        fens=_read_fens(args.positions)
        result=evaluate_directional_specificity(
            fens,
            baseline,
            rho=args.rho,
            magnitudes=args.magnitudes,
            controls=args.controls,
            seed=args.seed,
        )
        payload={
            "rho":result.rho,
            "seed":result.seed,
            "controls":result.controls,
            "settings":len(result.runs),
            "runs":[
                {
                    "magnitude":run.magnitude,
                    "aggregate":dict(run.aggregate),
                    "rows":[asdict(row) for row in run.rows],
                }
                for run in result.runs
            ],
        }
        receipt=build_receipt(command="specificity",arguments=vars(args),device=str(baseline.device),artifact_manifest=manifest,model_metadata=_model_metadata(weights),inputs=fens,results=payload,instability_count=0)
    elif args.command == "query-memory":
        fens=_read_fens(args.positions)
        result=evaluate_query_memory(fens, baseline, rho=args.rho, seed=args.seed)
        arms={
            "present":_retrieval_payload(result.present),
            "history":_retrieval_payload(result.history),
            "shuffled_history":_retrieval_payload(result.shuffled_history),
        }
        payload={
            "rho":result.rho,
            "seed":result.seed,
            "positions":result.positions,
            "arms":arms,
            "history_minus_present":{
                key: arms["history"][key] - arms["present"][key] for key in arms["history"]
            },
            "history_minus_shuffled":{
                key: arms["history"][key] - arms["shuffled_history"][key] for key in arms["history"]
            },
        }
        receipt=build_receipt(command="query-memory",arguments=vars(args),device=str(baseline.device),artifact_manifest=manifest,model_metadata=_model_metadata(weights),inputs=fens,results=payload,instability_count=0)
    elif args.command == "arena":
        openings=_read_fens(args.openings) if args.openings else [START_FEN]
        arena=play_paired_arena(ChessFlyPolicy(baseline),ChessFlyPolicy(stateping,forward_kwargs={"rho":args.rho,"kappa":args.kappa}),openings=openings,games=args.games,max_plies=args.max_plies,seed=args.seed)
        payload=asdict(arena)
        receipt=build_receipt(command="arena",arguments=vars(args),device=str(baseline.device),artifact_manifest=manifest,model_metadata=_model_metadata(weights),inputs=openings,results=payload,instability_count=arena.forfeits)
    else:
        raise AssertionError(args.command)
    output=_output_path(args.command,args.output); write_receipt(output,receipt); print(output)
    return 0


__all__ = ["build_parser", "default_sweep", "main"]
