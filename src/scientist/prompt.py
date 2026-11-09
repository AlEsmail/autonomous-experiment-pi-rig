# ABOUTME: Prompt builder and proposal parser for the scientist LLM client -- formats rig
# ABOUTME: status/history into the reply contract and extracts validated dose proposals.

import json
from dataclasses import dataclass

from scientist.settings import ScientistConfig
from shared.color import delta_e


class ProposalParseError(Exception):
    pass


@dataclass(frozen=True)
class Proposal:
    dose_a_ml: float
    dose_b_ml: float
    rationale: str


def build_prompt(status: dict, history: dict, cfg: ScientistConfig, notes: list[str]) -> str:
    """Prompt structure (wording is placeholder; fields are the contract): target hex +
    lambda + cycle budget/used from status; dose cap hint; the experiment table from
    history["cycles"] (n, dose_a_ml, dose_b_ml, measured_lab, blank_lab, delta_e, score,
    aborted); blank drift = each blank_lab's delta_e against the first completed cycle's
    blank_lab (spec: "full experiment table including blank drift"); visible events
    (kind + detail); the notes lines verbatim; and the reply contract: exactly one JSON
    object {"dose_a_ml": float, "dose_b_ml": float, "rationale": str}."""

    run = status.get("run", {})
    target_hex = status.get("target_hex", "")
    cycle_budget = run.get("cycle_budget", 0)
    cycles_used = run.get("cycles", 0)
    lam = status.get("lambda", 0.0)

    lines: list[str] = []
    lines.append(f"Target color: {target_hex}")
    lines.append(f"Lambda (score weight on dye ml): {lam}")
    lines.append(f"Cycle budget: {cycle_budget}, cycles used: {cycles_used}")
    lines.append(f"Dose cap hint: {cfg.dose_cap_hint_ml} ml")
    lines.append("")

    lines.append("Experiment history:")
    cycles = history.get("cycles", [])

    first_blank = None
    for row in cycles:
        if row.get("aborted") is None and row.get("blank_lab") is not None:
            first_blank = row["blank_lab"]
            break

    headers = [
        "n",
        "dose_a_ml",
        "dose_b_ml",
        "measured_lab",
        "blank_lab",
        "delta_e",
        "lambda",
        "score",
        "aborted",
        "blank_drift",
    ]
    lines.append(" | ".join(headers))

    for row in cycles:
        values = [str(row.get(h, "")) for h in headers[:-1]]
        drift = ""
        if first_blank is not None and row.get("blank_lab") is not None:
            drift = f"{delta_e(row['blank_lab'], first_blank):.2f}"
        values.append(drift)
        lines.append(" | ".join(values))

    lines.append("")
    lines.append("Visible events:")
    for event in history.get("events", []):
        lines.append(f"- {event.get('kind', '')}: {event.get('detail', '')}")

    if notes:
        lines.append("")
        lines.append("Notes:")
        for note in notes:
            lines.append(f"- {note}")

    lines.append("")
    lines.append("Reply with exactly one JSON object:")
    lines.append('{"dose_a_ml": float, "dose_b_ml": float, "rationale": str}')

    return "\n".join(lines)


def parse_proposal(text: str) -> Proposal:
    """First balanced {...} block in the reply (fences/prose tolerated), json-parsed.
    dose_a_ml/dose_b_ml must be non-negative int/float (bool excluded); rationale a
    non-empty str. Anything else -> ProposalParseError naming what failed."""

    start = text.find("{")
    if start == -1:
        raise ProposalParseError("no JSON object found in reply")

    depth = 0
    end = -1
    for i in range(start, len(text)):
        char = text[i]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end == -1:
        raise ProposalParseError("no balanced JSON object found in reply")

    json_text = text[start:end + 1]
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ProposalParseError(f"invalid JSON object: {exc}") from exc

    if not isinstance(data, dict):
        raise ProposalParseError("parsed JSON is not an object")

    for field in ("dose_a_ml", "dose_b_ml", "rationale"):
        if field not in data:
            raise ProposalParseError(f"missing field: {field}")

    for field in ("dose_a_ml", "dose_b_ml"):
        value = data[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ProposalParseError(f"{field} must be a non-negative number")
        if value < 0:
            raise ProposalParseError(f"{field} must be non-negative")

    rationale = data["rationale"]
    if not isinstance(rationale, str) or rationale == "":
        raise ProposalParseError("rationale must be a non-empty string")

    return Proposal(
        dose_a_ml=float(data["dose_a_ml"]),
        dose_b_ml=float(data["dose_b_ml"]),
        rationale=rationale,
    )
