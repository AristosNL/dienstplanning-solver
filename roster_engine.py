"""
Generieke CP-SAT roster-engine.

Eén engine voor alle planningsproblemen. De weekdienst-planning is een
speciaal geval: slots = werkdagen, demand = 1, kandidaten = de dienst-dokters.
Dezelfde engine vult straks de dagelijkse planning: andere slots (ochtend/middag
x activiteit), andere demands, andere skills. Niets in deze laag is hardcoded
naar 'dokters' of 'diensten'.

Model:
  beslissingsvariabele x[staff, slot] in {0,1}  -> medewerker werkt dat slot
  harde constraints  : coverage (demand), beschikbaarheid, skill, contracturen
  zachte constraints : fairness, continuiteit, voorkeur-vrij  (gewogen straf)
  objective          : minimaliseer som van gewogen strafpunten
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from ortools.sat.python import cp_model


class Period(str, Enum):
    FULL_DAY = "FULL_DAY"   # dienst
    AM = "AM"               # dagelijkse planning, ochtend
    PM = "PM"               # dagelijkse planning, middag


class Avail(str, Enum):
    AVAILABLE = "AVAILABLE"
    VACATION = "VACATION"        # hard: niet inplannen
    COURSE = "COURSE"            # hard: niet inplannen
    MANDATORY_OFF = "MANDATORY_OFF"  # hard: vaste vrije dag
    PREFER_OFF = "PREFER_OFF"    # zacht: liever niet


@dataclass(frozen=True)
class Staff:
    id: str
    name: str
    role: str
    skills: frozenset[str]          # welke activiteiten/diensten mag deze persoon
    contract_min_per_period: int = 10**9   # ruimte voor contracturen-limiet later
    carry_in: int = 0               # tellerstand uit het saldo-grootboek


@dataclass(frozen=True)
class Slot:
    id: str
    date: str                       # ISO yyyy-mm-dd
    period: Period
    required_skill: str             # welke skill is nodig om dit slot te vullen
    demand: int = 1                 # hoeveel mensen dit slot nodig heeft
    week: int = 0                   # voor continuiteit binnen-week
    seq: int = 0                    # volgorde binnen de week (ma=0..vr=4)


@dataclass
class SoftWeights:
    fairness: int = 10
    continuity: int = 3
    prefer_off: int = 1


@dataclass
class Result:
    feasible: bool
    status: str
    assignments: dict[str, list[str]] = field(default_factory=dict)  # slot_id -> [staff_id]
    totals: dict[str, int] = field(default_factory=dict)             # staff_id -> carry_in + nieuw
    objective: int = 0


class RosterEngine:
    def __init__(self, staff: list[Staff], slots: list[Slot],
                 availability: dict[tuple[str, str], Avail],
                 weights: SoftWeights | None = None):
        self.staff = staff
        self.slots = slots
        # availability key = (staff_id, date) -> status; ontbreekt = AVAILABLE
        self.availability = availability
        self.w = weights or SoftWeights()
        self.m = cp_model.CpModel()
        self.x: dict[tuple[str, str], cp_model.IntVar] = {}
        self.penalties: list[cp_model.LinearExpr] = []

    # ---- kandidaat-filter: harde uitsluitingen -------------------------------
    def _is_candidate(self, s: Staff, slot: Slot) -> bool:
        if slot.required_skill not in s.skills:
            return False
        a = self.availability.get((s.id, slot.date), Avail.AVAILABLE)
        if a in (Avail.VACATION, Avail.COURSE, Avail.MANDATORY_OFF):
            return False
        return True

    def build(self):
        # beslissingsvariabelen alleen voor toegestane (staff, slot)-paren
        for slot in self.slots:
            for s in self.staff:
                if self._is_candidate(s, slot):
                    self.x[(s.id, slot.id)] = self.m.NewBoolVar(f"x_{s.id}_{slot.id}")

        self._coverage()
        self._one_slot_per_period()
        self._fairness()
        self._continuity()
        self._prefer_off()
        self.m.Minimize(sum(self.penalties))

    # ---- HARD: elk slot krijgt precies (demand) mensen -----------------------
    def _coverage(self):
        for slot in self.slots:
            vars_ = [self.x[(s.id, slot.id)] for s in self.staff
                     if (s.id, slot.id) in self.x]
            # exact demand; bij infeasibility meldt de solver dat (ondergedekt)
            self.m.Add(sum(vars_) == slot.demand)

    # ---- HARD: niemand twee slots op hetzelfde dagdeel -----------------------
    def _one_slot_per_period(self):
        by_key: dict[tuple[str, str, Period], list] = {}
        for slot in self.slots:
            for s in self.staff:
                if (s.id, slot.id) in self.x:
                    by_key.setdefault((s.id, slot.date, slot.period), []).append(
                        self.x[(s.id, slot.id)])
        for vars_ in by_key.values():
            if len(vars_) > 1:
                self.m.Add(sum(vars_) <= 1)

    # ---- ZACHT: gelijke verdeling incl. saldo-grootboek ----------------------
    def _fairness(self):
        if self.w.fairness <= 0:
            return
        totals = {}
        for s in self.staff:
            assigned = [self.x[(s.id, slot.id)] for slot in self.slots
                        if (s.id, slot.id) in self.x]
            t = self.m.NewIntVar(0, len(self.slots) + s.carry_in, f"tot_{s.id}")
            self.m.Add(t == s.carry_in + sum(assigned))
            totals[s.id] = t
        lo = self.m.NewIntVar(0, 10**6, "min_total")
        hi = self.m.NewIntVar(0, 10**6, "max_total")
        self.m.AddMinEquality(lo, list(totals.values()))
        self.m.AddMaxEquality(hi, list(totals.values()))
        spread = self.m.NewIntVar(0, 10**6, "spread")
        self.m.Add(spread == hi - lo)
        self.penalties.append(self.w.fairness * spread)
        self._totals = totals

    # ---- ZACHT: continuiteit binnen de week (zo min mogelijk wisselingen) ----
    def _continuity(self):
        if self.w.continuity <= 0:
            return
        # groepeer per week, sorteer op seq, straf wissel tussen opeenvolgende dagen
        weeks: dict[int, list[Slot]] = {}
        for slot in self.slots:
            weeks.setdefault(slot.week, []).append(slot)
        for wk, group in weeks.items():
            group = sorted(group, key=lambda s: s.seq)
            for a, b in zip(group, group[1:]):
                same = self.m.NewBoolVar(f"same_{a.id}_{b.id}")
                # same == 1  <=>  dezelfde persoon op slot a en b
                ands = []
                for s in self.staff:
                    if (s.id, a.id) in self.x and (s.id, b.id) in self.x:
                        both = self.m.NewBoolVar(f"both_{s.id}_{a.id}_{b.id}")
                        self.m.AddBoolAnd([self.x[(s.id, a.id)],
                                           self.x[(s.id, b.id)]]).OnlyEnforceIf(both)
                        self.m.AddBoolOr([self.x[(s.id, a.id)].Not(),
                                          self.x[(s.id, b.id)].Not()]).OnlyEnforceIf(both.Not())
                        ands.append(both)
                if ands:
                    self.m.AddMaxEquality(same, ands)
                else:
                    self.m.Add(same == 0)
                switch = self.m.NewBoolVar(f"switch_{a.id}_{b.id}")
                self.m.Add(same + switch == 1)
                self.penalties.append(self.w.continuity * switch)

    # ---- ZACHT: voorkeur-vrij ------------------------------------------------
    def _prefer_off(self):
        if self.w.prefer_off <= 0:
            return
        for slot in self.slots:
            for s in self.staff:
                if (s.id, slot.id) in self.x:
                    if self.availability.get((s.id, slot.date)) == Avail.PREFER_OFF:
                        self.penalties.append(self.w.prefer_off * self.x[(s.id, slot.id)])

    def solve(self, max_seconds: float = 10.0) -> Result:
        self.build()
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = max_seconds
        st = solver.Solve(self.m)
        status = solver.StatusName(st)
        if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return Result(feasible=False, status=status)
        assignments: dict[str, list[str]] = {}
        for slot in self.slots:
            assignments[slot.id] = [s.id for s in self.staff
                                    if (s.id, slot.id) in self.x
                                    and solver.Value(self.x[(s.id, slot.id)]) == 1]
        totals = {}
        for s in self.staff:
            new = sum(1 for slot in self.slots
                      if (s.id, slot.id) in self.x
                      and solver.Value(self.x[(s.id, slot.id)]) == 1)
            totals[s.id] = s.carry_in + new
        return Result(feasible=True, status=status, assignments=assignments,
                      totals=totals, objective=int(solver.ObjectiveValue()))
