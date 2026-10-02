"""Same-present, fixed-ping assay of retained settling history.

This is an experimental input schedule, not a change to published ChessFly.
Continuation targets are the model's own unmodified next settling transitions.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import chess
import torch

from .encoding import ACTION_INDEX, canonical_fen, encode_fen, mirror_uci
from .orthogonal import orthogonal_history_readout, shuffled_orthogonal_control
from .retrieval_diagnostics import center_rows

EPS = 1e-12
TIE_ATOL = 1e-6


def scheduled_activity(model, features: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Run the frozen recurrence on [steps, batch, 780] sensory snapshots."""
    inputs = torch.as_tensor(features, dtype=torch.float32, device=model.device)
    if inputs.ndim != 3 or inputs.shape[0] != model.weights.steps or inputs.shape[2] != 780:
        raise ValueError("features must have shape [model steps, batch, 780]")
    if inputs.shape[1] < 1 or not torch.isfinite(inputs).all():
        raise ValueError("features must contain a nonempty batch of finite values")
    with torch.inference_mode():
        _, flat_drive = model._prepare(inputs.flatten(0, 1))
        drive = flat_drive.reshape(inputs.shape[0], inputs.shape[1], -1)
        hidden = torch.zeros_like(drive[0])
        states = []
        for step in range(model.weights.steps):
            recurrent = torch.sparse.mm(model.matrix, hidden.T).T
            pre = (recurrent + drive[step]) * model.tensors["scale"][step] + model.tensors["shift"][step]
            hidden = (1.0 - model.weights.alpha) * hidden + model.weights.alpha * torch.relu(pre)
            if not torch.isfinite(hidden).all():
                raise RuntimeError(f"non-finite scheduled activity at step {step + 1}")
            states.append(hidden.clone())
        return tuple(states)


def paired_continuation_scores(responses: torch.Tensor, targets: torch.Tensor) -> dict:
    """Score both history responses against both held-out target directions.

Targets indistinguishable under cosine, or zero response/target directions,
remain in the raw matrix but do not count as identifiable classification.
"""
    if responses.ndim != 2 or targets.shape != responses.shape or responses.shape[0] != 2:
        raise ValueError("responses and targets must share shape [2, features]")
    if responses.shape[1] < 1 or not torch.isfinite(responses).all() or not torch.isfinite(targets).all():
        raise ValueError("responses and targets must be nonempty and finite")
    response = responses.double()
    target = targets.double()
    response_norm = torch.linalg.vector_norm(response, dim=1)
    target_norm = torch.linalg.vector_norm(target, dim=1)
    normalized_response = response / response_norm.clamp_min(EPS)[:, None]
    normalized_target = target / target_norm.clamp_min(EPS)[:, None]
    scores = normalized_response @ normalized_target.T
    target_separation = float(torch.linalg.vector_norm(normalized_target[0] - normalized_target[1]).item())
    response_separation = float((
        torch.linalg.vector_norm(response[0] - response[1]) / response_norm.max().clamp_min(EPS)
    ).item())
    identifiable = bool((response_norm > EPS).all() and (target_norm > EPS).all()) and target_separation > TIE_ATOL
    native = scores.diagonal()
    crossed = torch.stack((scores[0, 1], scores[1, 0]))
    margins = native - crossed
    credits = torch.where(margins > TIE_ATOL, 1.0, torch.where(margins < -TIE_ATOL, 0.0, 0.5))
    return {
        "identifiable": identifiable,
        "scores": scores.detach().cpu().tolist(),
        "margins": margins.detach().cpu().tolist(),
        "accuracy": float(credits.mean().item()) if identifiable else None,
        "mean_correct_margin": float(margins.mean().item()) if identifiable else None,
        "mean_native_alignment": float(native.mean().item()) if identifiable else None,
        "response_separation": response_separation,
        "target_direction_separation": target_separation,
        "target_norms": target_norm.detach().cpu().tolist(),
        "response_norms": response_norm.detach().cpu().tolist(),
    }


@dataclass(frozen=True, slots=True)
class HistoryLensCase:
    id: str
    present: str
    cue_a: str
    cue_b: str


def _encoded(fen: str) -> torch.Tensor:
    board = chess.Board(fen)
    if not board.is_valid() or board.is_game_over():
        raise ValueError("case FENs must be valid nonterminal chess positions")
    canonical, _ = canonical_fen(fen)
    return torch.tensor(encode_fen(canonical), dtype=torch.float32)


def _validate_cases(cases) -> tuple[HistoryLensCase, ...]:
    cases = tuple(cases)
    if not cases:
        raise ValueError("cases must be nonempty")
    ids = set()
    for case in cases:
        if not isinstance(case.id, str) or not case.id.strip():
            raise ValueError("case id must be a nonempty string")
        if case.id in ids:
            raise ValueError(f"duplicate case id: {case.id}")
        ids.add(case.id)
        _encoded(case.present)
        if len(tuple(chess.Board(case.present).legal_moves)) < 2:
            raise ValueError("present positions require at least two legal moves")
        if torch.equal(_encoded(case.cue_a), _encoded(case.cue_b)):
            raise ValueError("cue encodings must differ; FEN metadata alone is insufficient")
    return cases


def load_history_cases(path: str | Path) -> tuple[HistoryLensCase, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("case file must contain a cases array")
    try:
        cases = tuple(HistoryLensCase(**row) for row in rows)
    except (TypeError, KeyError) as exc:
        raise ValueError("each case requires id, present, cue_a and cue_b") from exc
    return _validate_cases(cases)


def paired_history_activity(model, cues: torch.Tensor, present: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """A,B and B,A prefixes, then identical present; third row erases history."""
    if cues.shape != (2, 780) or present.shape != (780,):
        raise ValueError("cues require [2,780] and present [780]")
    if model.weights.steps < 4:
        raise ValueError("history lens requires at least four settling steps")
    inputs = present[None, None, :].repeat(model.weights.steps, 3, 1)
    inputs[0, :2] = cues
    inputs[1, :2] = cues.flip(0)
    return scheduled_activity(model, inputs)


def fixed_history_ping(activity, model, *, rho: float = 0.75, prefix_steps: int = 3):
    """Construct the common query solely from a present-only causal prefix."""
    if prefix_steps not in (2, 3) or len(activity) < prefix_steps or any(row.shape[0] != 3 for row in activity[:prefix_steps]):
        raise ValueError("ping construction requires two or three three-row prefix states")
    return orthogonal_history_readout(
        tuple(row[2:3] for row in activity[:prefix_steps]),
        rho=rho,
        readout_indices=model.readout_index,
    )


def _odd_association(model, states, directions, magnitude):
    trace = model.decode_readout_trace(torch.cat((states + magnitude * directions, states - magnitude * directions)))
    plus, minus = trace.association.chunk(2, dim=0)
    return (plus - minus) / (2.0 * magnitude)


def fixed_ping_association(model, states, ping, *, magnitude: float):
    if states.ndim != 2 or ping.shape != (1, states.shape[1]):
        raise ValueError("one fixed ping must have shape [1, receiver features]")
    magnitude = float(magnitude)
    if not math.isfinite(magnitude) or magnitude <= 0:
        raise ValueError("magnitude must be positive and finite")
    return _odd_association(model, states, ping.expand_as(states), magnitude)


def _project(association, model, legal_indices):
    return {
        "association": association,
        "value": center_rows(association @ model.tensors["value.weight"].T),
        "policy": center_rows(association @ model.tensors["policy.weight"].index_select(0, legal_indices).T),
    }


def _mean_margin(metric):
    # Include zero-response controls as zero information, on the same target set.
    return mean(metric["margins"])


def _aggregate(rows, geometry, control_count):
    valid = [row[geometry] for row in rows if row[geometry]["retained"]["identifiable"]]
    result = {"identifiable_cases": len(valid), "queries": 2 * len(valid)}
    if not valid:
        result.update({name: None for name in (
            "accuracy", "mean_correct_margin", "mean_native_alignment",
            "mean_response_separation", "erased_margin", "swapped_margin",
            "linear_margin", "retained_minus_erased", "control_margin_mean", "control_margin_percentile",
        )})
        result["control_margin_means"] = [None] * control_count
        return result
    retained_margin = mean(packet["retained"]["mean_correct_margin"] for packet in valid)
    erased_margin = mean(_mean_margin(packet["erased"]) for packet in valid)
    control_margins = [mean(_mean_margin(packet["controls"][j]) for packet in valid) for j in range(control_count)]
    less = sum(margin < retained_margin - TIE_ATOL for margin in control_margins)
    equal = sum(abs(margin - retained_margin) <= TIE_ATOL for margin in control_margins)
    result.update({
        "accuracy": mean(packet["retained"]["accuracy"] for packet in valid),
        "mean_correct_margin": retained_margin,
        "mean_native_alignment": mean(packet["retained"]["mean_native_alignment"] for packet in valid),
        "mean_response_separation": mean(packet["retained"]["response_separation"] for packet in valid),
        "erased_margin": erased_margin,
        "swapped_margin": mean(_mean_margin(packet["swapped"]) for packet in valid),
        "linear_margin": mean(_mean_margin(packet["linear"]) for packet in valid),
        "retained_minus_erased": retained_margin - erased_margin,
        "control_margin_mean": mean(control_margins),
        "control_margin_percentile": (less + 0.5 * equal + 0.5) / (control_count + 1.0),
        "control_margin_means": control_margins,
    })
    return result


def evaluate_history_lens(
    cases, model, *, rho=0.75, delays=(2,), magnitudes=(0.5, 1.0, 2.0), controls=32, seed=0, ping_steps=3,
) -> dict:
    """Read fixed pings through two histories and score withheld continuations."""
    cases = _validate_cases(cases)
    delays = tuple(delays)
    magnitudes = tuple(float(a) for a in magnitudes)
    if not math.isfinite(rho) or not 0 <= rho < 1:
        raise ValueError("rho must be finite and satisfy 0 <= rho < 1")
    if not delays or any(not isinstance(d, int) or not 1 <= d <= model.weights.steps - 3 for d in delays):
        raise ValueError("delays must leave an independent next settling step")
    if not isinstance(ping_steps, int) or ping_steps not in (2, 3):
        raise ValueError("ping_steps must be 2 or 3")
    if any(2 + delay <= ping_steps for delay in delays):
        raise ValueError("delayed query must arrive after ping construction")
    if len(set(delays)) != len(delays) or len(set(magnitudes)) != len(magnitudes):
        raise ValueError("delays and magnitudes must be unique")
    if not magnitudes or any(not math.isfinite(a) or a <= 0 for a in magnitudes):
        raise ValueError("magnitudes must be positive and finite")
    if not isinstance(controls, int) or controls < 1:
        raise ValueError("controls must be a positive integer")
    runs = [{"delay": d, "magnitude": a, "cases": []} for d in delays for a in magnitudes]
    with torch.inference_mode():
        for case_index, case in enumerate(cases):
            cues = torch.stack((_encoded(case.cue_a), _encoded(case.cue_b)))
            activity = paired_history_activity(model, cues, _encoded(case.present))
            reference = fixed_history_ping(activity, model, rho=rho, prefix_steps=ping_steps)
            ping = reference.orthogonal
            directions = torch.cat((ping, *(
                shuffled_orthogonal_control(reference.current, ping, seed=int(seed) + 1009 * case_index + j + 1)
                for j in range(controls)
            )))
            board = chess.Board(case.present)
            _, mirrored = canonical_fen(case.present)
            legal_indices = torch.tensor([
                ACTION_INDEX[mirror_uci(move.uci()) if mirrored else move.uci()]
                for move in board.legal_moves
            ], dtype=torch.long, device=model.device)
            linear_association = (ping @ model.tensors["decoder.weight"].T).repeat(2, 1)
            linear = _project(linear_association, model, legal_indices)
            for delay in delays:
                query_index = 1 + delay
                states = activity[query_index].index_select(1, model.readout_index)
                next_states = activity[query_index + 1][:2].index_select(1, model.readout_index)
                base = model.decode_readout_trace(states[:2]).association
                future = model.decode_readout_trace(next_states).association
                target = _project(future - base, model, legal_indices)
                probe_states = states[:2].repeat(controls + 1, 1)
                probe_directions = directions.repeat_interleave(2, dim=0)
                for run in (r for r in runs if r["delay"] == delay):
                    association = _odd_association(model, probe_states, probe_directions, run["magnitude"])
                    projected = _project(association, model, legal_indices)
                    erased_association = fixed_ping_association(model, states[2:3].repeat(2, 1), ping, magnitude=run["magnitude"])
                    erased = _project(erased_association, model, legal_indices)
                    row = {
                        "id": case.id, "query_step": query_index + 1, "target_step": query_index + 2,
                        "ping_norm": float(torch.linalg.vector_norm(ping).item()),
                        "hidden_separation": float(torch.linalg.vector_norm(activity[query_index][0] - activity[query_index][1]).item()),
                        "readout_separation": float(torch.linalg.vector_norm(states[0] - states[1]).item()),
                    }
                    for geometry in ("association", "value", "policy"):
                        responses = projected[geometry].reshape(controls + 1, 2, -1)
                        row[geometry] = {
                            "retained": paired_continuation_scores(responses[0], target[geometry]),
                            "erased": paired_continuation_scores(erased[geometry], target[geometry]),
                            "swapped": paired_continuation_scores(responses[0].flip(0), target[geometry]),
                            "linear": paired_continuation_scores(linear[geometry], target[geometry]),
                            "controls": [paired_continuation_scores(response, target[geometry]) for response in responses[1:]],
                        }
                    run["cases"].append(row)
    for run in runs:
        run["aggregate"] = {g: _aggregate(run["cases"], g, controls) for g in ("association", "value", "policy")}
    primary_delay = 2 if ping_steps == 3 else 1
    primary_run = next((r for r in runs if r["delay"] == primary_delay and r["magnitude"] == 1.0), None)
    primary = {"delay": primary_delay, "magnitude": 1.0, "geometry": "centered_legal_policy", "minimum_cases": 12}
    if primary_run is None:
        primary["status"] = "not_run"
    else:
        stats = primary_run["aggregate"]["policy"]
        primary["metrics"] = stats
        if stats["identifiable_cases"] < 12:
            primary["status"] = "inconclusive"
        else:
            criteria = {
                "native_margin": stats["mean_correct_margin"] > TIE_ATOL,
                "accuracy": stats["accuracy"] >= 0.75,
                "positive_alignment": stats["mean_native_alignment"] > 0,
                "beats_erased": stats["retained_minus_erased"] > TIE_ATOL,
                "matched_control_percentile": stats["control_margin_percentile"] >= 0.95,
            }
            primary["criteria"] = criteria
            primary["status"] = "passed" if all(criteria.values()) else "failed"
    return {
        "protocol": f"same-present-history-lens-v{2 if ping_steps == 3 else 1}", "rho": float(rho), "seed": int(seed),
        "ping_steps": ping_steps,
        "controls": controls, "case_count": len(cases), "primary": primary, "runs": runs,
        "target": "next unmodified settling transition; internal prediction, not chess optimality",
        "ping_source": f"present-only prefix steps 1–{ping_steps}; identical across histories; no future target access",
    }
