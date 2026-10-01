"""
Prompt templates for the three LLM-driven nodes:
  1. Severity classifier  — chain-of-thought, outputs TriageResult JSON
  2. Root cause analyser  — few-shot grounded, outputs a narrative string
  3. Remediation planner  — role + hard constraints, outputs RemediationPlan JSON

All templates use Jinja2 and output JSON that is validated against the Pydantic
schemas defined here.  Validation failure → retry signal, not a silent patch.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, field_validator


# ── Pydantic output schemas ───────────────────────────────────────────────────

class TriageResult(BaseModel):
    pipeline: str
    severity: Literal["P1", "P2", "P3"]
    error_type: str
    sla_at_risk: bool
    downstream_count: int
    reasoning: str   # CoT trace — visible to human approver


class RemediationStep(BaseModel):
    step_number: int
    action: str
    sql_or_cmd: Optional[str] = None
    risk_level: Literal["low", "medium", "high"]
    reversible: bool


class RemediationPlan(BaseModel):
    incident_id: str
    root_cause: str
    similar_incidents: List[str]   # INC-XXX references
    steps: List[RemediationStep]
    estimated_mins: int
    requires_approval: bool        # always True when any step has sql_or_cmd


    @field_validator("steps")
    @classmethod
    def at_least_one_step(cls, v: list) -> list:
        if not v:
            raise ValueError("Remediation plan must have at least one step.")
        return v

    @field_validator("requires_approval", mode="before")
    @classmethod
    def enforce_approval_for_writes(cls, v: bool, info) -> bool:  # noqa: ARG002
        # requires_approval must be True whenever any step has sql_or_cmd
        return True  # We enforce this unconditionally — any plan touching data needs human sign-off


# ── PROMPT 1: Severity Classifier (CoT) ──────────────────────────────────────
SEVERITY_TMPL = """\
You are an on-call data engineering lead at Atlas Retail Group.
A new pipeline incident has been reported. Your job is to classify its severity
using a structured chain-of-thought, then output a JSON object matching the schema below.

## Severity rules (apply in order — first match wins)
| Severity | Condition |
|----------|-----------|
| P1       | Pipeline criticality is "critical" OR SLA window is breached/imminent (< 1 h remaining) OR data loss / compliance violation detected |
| P1       | Downstream feeds 2+ critical pipelines |
| P2       | SLA at risk (> 50% of window elapsed) OR data correctness issue with no compliance flag |
| P3       | Cosmetic / non-blocking / low-criticality pipeline |

## Incident context
- **Pipeline:** {{ pipeline_name }}
- **Criticality:** {{ criticality }}
- **SLA window:** {{ sla_hours }} hours
- **Error:** {{ error_type }}
- **Parsed log details:** {{ parsed_incident | tojson }}
- **Downstream pipelines:** {{ downstream | join(', ') }}
- **SQL diagnostic result:** {{ sql_result | tojson }}
- **Similar past incidents:** {{ similar_incidents | map(attribute='incident') | list | map(attribute='severity') | list | join(', ') if similar_incidents else 'none retrieved' }}

## Reasoning steps (complete all before classifying)
1. Is this pipeline marked "critical"?  What are the downstream impacts?
2. How much of the SLA window has elapsed?  Is a breach imminent?
3. Is there evidence of data loss, duplication, PII exposure, or compliance risk?
4. What severity do the most similar past incidents carry?
5. Final classification: P1 / P2 / P3 and why.

## Output schema (JSON only — no markdown fences, no extra keys)
{
  "pipeline": "<pipeline name>",
  "severity": "P1"|"P2"|"P3",
  "error_type": "<short error label>",
  "sla_at_risk": true|false,
  "downstream_count": <integer>,
  "reasoning": "<your full CoT from steps 1-5, single string>"
}

Think through all five steps, then output ONLY the JSON object.
"""


# ── PROMPT 2: Root Cause Analyser (Few-shot) ─────────────────────────────────
ROOT_CAUSE_TMPL = """\
You are a senior data engineer diagnosing a pipeline failure.
Using the log evidence, SQL diagnostic results, and similar past incidents,
identify the most likely root cause in 2–3 sentences.
Cite actual values from the evidence (row counts, error messages, column names).
Do NOT fabricate causes not supported by the data.

## Few-shot examples

### Example 1
Log: "NullPointerException in transform step | 0 rows loaded from raw_sales | Expected: ~1.2M rows"
SQL result: row_counts shows raw_sales = 0 rows as of 2025-06-01 01:00
Similar incident: INC-001 — "Upstream source table loaded 0 rows due to failed CDC job"
Root cause: The raw_sales CDC job failed to deliver any rows (0 loaded vs ~1.2M expected),
causing a NullPointerException in the transform step — identical to INC-001 where the CDC
job required a restart and full-refresh rerun.

### Example 2
Log: "Schema mismatch: column 'loyalty_tier' not found in staging.customers"
SQL result: no schema-change table available; column absent in SELECT *
Similar incident: INC-002 — "Source team added column without notifying downstream"
Root cause: A new column (loyalty_tier) was added to the source CRM system without
coordinating with the downstream pipeline, causing a schema mismatch in the staging
table — matching the pattern in INC-002 where an ALTER TABLE and dbt model update resolved it.

## Current incident

**Pipeline:** {{ pipeline_name }}
**Parsed log details:**
{{ parsed_incident | tojson(indent=2) }}

**SQL diagnostic result:**
{{ sql_result | tojson(indent=2) }}

**Similar past incidents (top {{ similar_incidents | length }}):**
{% for r in similar_incidents %}
- {{ r.incident.id }}: {{ r.incident.error }} → {{ r.incident.root_cause }} ({{ r.incident.severity }}, resolved in {{ r.incident.resolved_mins }} min)
{% endfor %}
{% if not similar_incidents %}
- No strong precedent found in knowledge base.
{% endif %}

## Root cause (2–3 sentences, evidence-grounded, no fabrication):
"""


# ── PROMPT 3: Remediation Planner (Role + Constraints) ───────────────────────
PLAN_TMPL = """\
You are a senior data engineer at Atlas Retail Group writing a remediation plan
for a production pipeline incident.

## Hard constraints (non-negotiable)
1. NEVER propose an unguarded DROP, TRUNCATE, or unbounded DELETE (without a WHERE clause).
2. Every step that modifies data must have risk_level "medium" or "high" and reversible=false
   if it cannot be easily undone.
3. requires_approval MUST be true whenever any step contains sql_or_cmd.
4. Order steps from safest to most invasive.
5. Keep estimated_mins realistic — base it on the similar incidents' resolved_mins.
{% if human_feedback %}
6. **REVISION REQUIRED:** The previous plan was REJECTED.
   Human reviewer feedback: "{{ human_feedback }}"
   Address this feedback explicitly in the revised plan.
{% endif %}

## Incident summary
- **Incident ID:** {{ incident_id }}
- **Pipeline:** {{ pipeline_name }}
- **Severity:** {{ severity }}
- **Root cause:** {{ root_cause }}
- **Parsed log:** {{ parsed_incident | tojson }}

## Similar resolved incidents
{% for r in similar_incidents %}
- {{ r.incident.id }}: {{ r.incident.resolution }}
  Fix SQL: {{ r.incident.fix_sql }}
{% endfor %}
{% if not similar_incidents %}
- No strong precedent; use engineering judgment.
{% endif %}

## SQL diagnostic results
{{ sql_result | tojson(indent=2) }}

## Output schema (JSON only — no markdown fences)
{
  "incident_id": "{{ incident_id }}",
  "root_cause": "<concise root cause, one sentence>",
  "similar_incidents": ["INC-XXX", ...],
  "steps": [
    {
      "step_number": 1,
      "action": "<what to do>",
      "sql_or_cmd": "<SQL or shell command, or null>",
      "risk_level": "low"|"medium"|"high",
      "reversible": true|false
    }
    // ... more steps
  ],
  "estimated_mins": <integer>,
  "requires_approval": true
}

Output ONLY the JSON object.
"""
