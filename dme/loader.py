"""Read the case and the directory off disk. No database, per the brief."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from .clock import CENTRAL
from .model import Case, Patient, Supplier

DATA = Path(__file__).resolve().parent.parent / "data"


def read_data(name: str) -> str:
    """The fixture files, wherever they happen to live.

    On a normal machine they are on disk. Inside a Worker there is no
    filesystem, so the build step writes them into `_bundled_data.py` and this
    falls through to that. The rest of the loader does not need to know.
    """
    try:
        path = DATA / name
        if path.exists():
            return path.read_text(encoding="utf-8")
    except OSError:
        pass
    from ._bundled_data import FILES

    return FILES[name]


def load_suppliers(path: Path | None = None) -> dict[str, Supplier]:
    """The directory exactly as given: name, phone, address. Nothing else exists."""
    text = path.read_text(encoding="utf-8") if path else read_data("suppliers.csv")
    suppliers: dict[str, Supplier] = {}
    if True:
        for i, row in enumerate(csv.DictReader(text.splitlines()), start=1):
            supplier_id = f"s{i:02d}"
            suppliers[supplier_id] = Supplier(
                supplier_id=supplier_id,
                name=row["supplier_name"].strip(),
                phone=row["phone"].strip(),
                address=row["address"].strip(),
            )
    return suppliers


def suppliers_from_rows(rows: list[dict]) -> dict[str, Supplier]:
    """Build the directory from arbitrary rows -- the web form, or any other source."""
    suppliers: dict[str, Supplier] = {}
    for i, row in enumerate(rows, start=1):
        name = (row.get("name") or row.get("supplier_name") or "").strip()
        if not name:
            continue
        supplier_id = f"s{i:02d}"
        suppliers[supplier_id] = Supplier(
            supplier_id=supplier_id,
            name=name,
            phone=(row.get("phone") or "").strip(),
            address=(row.get("address") or "").strip(),
        )
    return suppliers


def parse_opened_at(value: str) -> datetime:
    """Accept a form's naive local time as well as a full ISO stamp.

    A browser datetime-local field has no zone. Everything downstream compares
    against business hours in Chicago, so a naive value means Chicago -- and
    saying so here is better than a TypeError four layers down.
    """
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=CENTRAL)


def build_case(payload: dict) -> tuple[Case, list[str]]:
    """A Case from a plain dict, so a case can come from a form as easily as a file.

    Eleanor is the fixture the brief supplies, not a special case in the code.
    Nothing below knows her name.
    """
    p = payload.get("patient", {})
    case = Case(
        case_id=payload.get("case_id") or "case-adhoc",
        patient=Patient(
            name=p.get("name") or "the patient",
            age=int(p.get("age") or 0),
            coverage=p.get("coverage") or "Original Medicare (Part B)",
            has_supplemental=bool(p.get("has_supplemental")),
            zip_code=str(p.get("zip_code") or p.get("zip") or ""),
            zip_is_assumed=bool(p.get("zip_is_assumed", p.get("zip_assumed", False))),
            phone=p.get("phone") or "",
        ),
        equipment=payload.get("equipment") or "Standard manual wheelchair",
        hcpcs=payload.get("hcpcs") or "K0001",
        pcp_name=payload.get("pcp_name") or "the ordering physician",
        pcp_practice=payload.get("pcp_practice") or "the practice",
        pcp_phone=payload.get("pcp_phone") or "",
        opened_at=parse_opened_at(payload["opened_at"]),
        suppliers=suppliers_from_rows(payload.get("suppliers") or []),
    )
    case.order.hcpcs = case.hcpcs
    return case, list(payload.get("assumptions") or [])


def default_payload(path: Path | None = None, suppliers_path: Path | None = None) -> dict:
    """The brief's own case, as a payload the form can be pre-filled from."""
    blob = json.loads(
        path.read_text(encoding="utf-8") if path else read_data("case_eleanor.json")
    )
    blob["suppliers"] = [
        {"name": s.name, "phone": s.phone, "address": s.address}
        for s in load_suppliers(suppliers_path).values()
    ]
    p = blob["patient"]
    p.setdefault("zip_code", p.get("zip", ""))
    p.setdefault("zip_is_assumed", p.get("zip_assumed", False))
    return blob


def load_case(path: Path | None = None, suppliers_path: Path | None = None) -> tuple[Case, list[str]]:
    blob = json.loads(
        path.read_text(encoding="utf-8") if path else read_data("case_eleanor.json")
    )
    p = blob["patient"]
    case = Case(
        case_id=blob["case_id"],
        patient=Patient(
            name=p["name"],
            age=p["age"],
            coverage=p["coverage"],
            has_supplemental=p["has_supplemental"],
            zip_code=p["zip_code"],
            zip_is_assumed=p["zip_is_assumed"],
            phone=p["phone"],
        ),
        equipment=blob["equipment"],
        hcpcs=blob["hcpcs"],
        pcp_name=blob["pcp_name"],
        pcp_practice=blob["pcp_practice"],
        pcp_phone=blob["pcp_phone"],
        opened_at=datetime.fromisoformat(blob["opened_at"]),
        suppliers=load_suppliers(suppliers_path),
    )
    case.order.hcpcs = blob["hcpcs"]
    return case, blob.get("assumptions", [])
