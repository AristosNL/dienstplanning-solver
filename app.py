"""
app.py — Dienstplanning solver-service (FastAPI + OR-Tools CP-SAT)

Endpoint:
  POST /solve-weekday   → berekent de doordeweekse dienstplanning
  GET  /health          → healthcheck

De service is bewust dun: hij vertaalt een verzoek naar de generieke
RosterEngine en geeft het resultaat per datum terug. Dezelfde engine
bedient later ook de dagplanning (ander endpoint, andere slots).

CORS staat open zodat de React-site (localhost of Netlify) hem mag aanroepen.
"""

from datetime import date, timedelta, datetime as dt
from io import BytesIO
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openpyxl

from roster_engine import RosterEngine, Staff, Slot, Period, Avail, SoftWeights

SKILL = "DIENST_WEEKDAG"
WD_CODES = ["ma", "di", "wo", "do", "vr", "za", "zo"]

app = FastAPI(title="Dienstplanning solver", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class DoctorIn(BaseModel):
    id: str
    carryIn: int = 0
    fixedOff: list[str] = []          # weekdagcodes, bv. ["wo"]
    preferOff: list[str] = []         # weekdagcodes, zacht
    absences: list[dict] = []         # [{"from":"2026-07-06","to":"2026-07-10","type":"vakantie"}]


class SolveReq(BaseModel):
    start: str                        # ISO-datum; wordt naar maandag uitgelijnd
    weeks: int = 4
    doctors: list[DoctorIn]
    weights: dict | None = None


@app.get("/health")
def health():
    return {"ok": True}


# ---------------------------------------------------------------------------
# /parse-requirements
# Accepteert OK- en/of PBK-Excel (multipart), retourneert geldige PLA-slots.
# Een PLA is geldig als: niet doorgestreept EN niet overgedragen ("PLA > X").
# Structuur retour: { requirements: { "2026-01-05": { "AM": ["OK"], "PM": [] } } }
# ---------------------------------------------------------------------------

_DAYS_NL = {"MAANDAG", "DINSDAG", "WOENSDAG", "DONDERDAG", "VRIJDAG"}
# Per dag-blok: (datum_kol, ochtend_kol, middag_kol) voor weken 1-4
_WEEK_COLS = [(2, 2, 3), (4, 4, 5), (6, 6, 7), (8, 8, 9)]


def _extract(content: bytes, kind: str, result: dict):
    """Parseer één Excel-bestand en voeg geldige PLA-requirements toe aan result."""
    wb_v = openpyxl.load_workbook(BytesIO(content), data_only=True)
    wb_f = openpyxl.load_workbook(BytesIO(content), data_only=False)

    for sheet in wb_v.sheetnames:
        wsv = wb_v[sheet]
        wsf = wb_f[sheet]

        for r in range(1, wsv.max_row + 1):
            a = wsv.cell(r, 1).value
            if not (a and str(a).strip().upper() in _DAYS_NL):
                continue

            # Verzamel rijen onder dit dag-anker tot de volgende dag
            block = []
            rr = r + 1
            while rr <= wsv.max_row:
                nxt = wsv.cell(rr, 1).value
                if nxt and str(nxt).strip().upper() in _DAYS_NL:
                    break
                block.append(rr)
                rr += 1

            for (dc, oc, mc) in _WEEK_COLS:
                dval = wsv.cell(r, dc).value
                if not isinstance(dval, dt):
                    continue
                date_str = dval.strftime("%Y-%m-%d")

                for (col, period) in [(oc, "AM"), (mc, "PM")]:
                    for br in block:
                        v = wsv.cell(br, col).value
                        if not (v and "PLA" in str(v).upper()):
                            continue
                        cell_f = wsf.cell(br, col)
                        struck = bool(cell_f.font and cell_f.font.strikethrough)
                        transfer = ">" in str(v)   # "PLA > NCH" = overgedragen
                        if struck or transfer:
                            continue
                        # Geldige requirement — voeg toe (dedupliceer)
                        slot = result.setdefault(date_str, {"AM": [], "PM": []})
                        if kind not in slot[period]:
                            slot[period].append(kind)


@app.post("/parse-requirements")
async def parse_requirements(
    ok_file:  UploadFile = File(None),
    pbk_file: UploadFile = File(None),
):
    result: dict = {}
    if ok_file  and ok_file.filename:
        _extract(await ok_file.read(),  "OK",  result)
    if pbk_file and pbk_file.filename:
        _extract(await pbk_file.read(), "PBK", result)

    count = sum(
        len(v.get("AM", [])) + len(v.get("PM", []))
        for v in result.values()
    )
    return {"requirements": result, "count": count}


@app.post("/solve-weekday")
def solve_weekday(req: SolveReq):
    start = date.fromisoformat(req.start)
    start = start - timedelta(days=start.weekday())   # uitlijnen op maandag

    # slots: ma-vr voor elke week
    slots = []
    for w in range(req.weeks):
        monday = start + timedelta(weeks=w)
        for i in range(5):
            d = monday + timedelta(days=i)
            slots.append(Slot(
                id=f"D{d.isoformat()}", date=d.isoformat(),
                period=Period.FULL_DAY, required_skill=SKILL,
                demand=1, week=w, seq=i,
            ))

    # staff + beschikbaarheid afleiden uit vaste vrije dagen / voorkeur / absenties
    staff, avail = [], {}
    for doc in req.doctors:
        staff.append(Staff(id=doc.id, name=doc.id, role="dokter",
                           skills=frozenset({SKILL}), carry_in=doc.carryIn))
        for slot in slots:
            wd = WD_CODES[date.fromisoformat(slot.date).weekday()]
            if wd in doc.fixedOff:
                avail[(doc.id, slot.date)] = Avail.MANDATORY_OFF
            elif wd in doc.preferOff:
                avail.setdefault((doc.id, slot.date), Avail.PREFER_OFF)
        for ab in doc.absences:
            f, t = ab.get("from"), ab.get("to")
            if not f or not t:
                continue
            kind = Avail.COURSE if ab.get("type") == "cursus" else Avail.VACATION
            for slot in slots:
                if f <= slot.date <= t:
                    avail[(doc.id, slot.date)] = kind

    w = req.weights or {}
    weights = SoftWeights(
        fairness=w.get("fairness", 10),
        continuity=w.get("continuity", 3),
        prefer_off=w.get("preferOff", 1),
    )

    res = RosterEngine(staff, slots, avail, weights).solve()
    if not res.feasible:
        return {"feasible": False, "status": res.status, "assignments": {}, "totals": {}, "objective": 0}

    assignments = {}
    for slot in slots:
        who = res.assignments.get(slot.id, [])
        if who:
            assignments[slot.date] = who[0]

    return {
        "feasible": True, "status": res.status,
        "assignments": assignments, "totals": res.totals, "objective": res.objective,
    }
