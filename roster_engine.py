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
    same_day_pair: int = 0   # regel 1: AM+PM van zelfde skill -> liefst zelfde persoon
    poli_split: int = 0      # regel 2: niemand Poli in AM en PM op dezelfde dag
    ok_fairness: int = 0     # regel: OK-diensten apart eerlijk verdelen binnen de week
    ok_skill: str = ""       # welke required_skill telt als "OK" voor ok_fairness


@dataclass
class Result:
    feasible: bool
    status: str
    assignments: dict[str, list[str]] = field(default_factory=dict)  # slot_id -> [staff_id]
    totals: dict[str, int] = field(default_factory=dict)             # staff_id -> carry_in + nieuw
    objective: int = 0


class RosterEngine:
    UNCOV_PENALTY = 1000   # straf per ongedekt slot bij partial_coverage

    def __init__(self, staff: list[Staff], slots: list[Slot],
                 availability: dict[tuple[str, str], Avail],
                 weights: SoftWeights | None = None,
                 partial_coverage: bool = False,
                 pair_skill: str | None = None,
                 split_skill: str | None = None):
        self.staff = staff
        self.slots = slots
        self.availability = availability
        self.w = weights or SoftWeights()
        self.partial_coverage = partial_coverage
        self.pair_skill = pair_skill      # skill waarvoor AM+PM zelfde persoon (regel 1)
        self.split_skill = split_skill    # skill die niet AM+PM bij één persoon mag (regel 2)
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
        self._ok_fairness()
        self._continuity()
        self._prefer_off()
        self._same_day_pair()
        self._poli_split()
        self.m.Minimize(sum(self.penalties))

    # ---- HARD/SOFT: elk slot krijgt (demand) mensen -------------------------
    def _coverage(self):
        for slot in self.slots:
            vars_ = [self.x[(s.id, slot.id)] for s in self.staff
                     if (s.id, slot.id) in self.x]
            if not vars_:
                continue          # geen kandidaten: onoplosbaar, skip
            if self.partial_coverage:
                # Soft: probeer demand te halen, straf elk tekort zwaar
                self.m.Add(sum(vars_) <= slot.demand)
                shortfall = self.m.NewIntVar(0, slot.demand, f"short_{slot.id}")
                self.m.Add(shortfall == slot.demand - sum(vars_))
                self.penalties.append(self.UNCOV_PENALTY * shortfall)
            else:
                # Hard: exact demand (klassieker gedrag)
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

    def _ok_fairness(self):
        """Verdeel specifiek de OK-slots zo gelijk mogelijk binnen de week.
        Los van de algemene fairness (die telt alle types samen, waardoor
        veel Poli een scheve OK-verdeling kan maskeren)."""
        if self.w.ok_fairness <= 0 or not self.w.ok_skill:
            return
        ok_slots = [s for s in self.slots if s.required_skill == self.w.ok_skill]
        if not ok_slots:
            return
        totals = []
        for s in self.staff:
            assigned = [self.x[(s.id, slot.id)] for slot in ok_slots
                        if (s.id, slot.id) in self.x]
            if not assigned:
                continue
            t = self.m.NewIntVar(0, len(ok_slots), f"oktot_{s.id}")
            self.m.Add(t == sum(assigned))
            totals.append(t)
        if len(totals) < 2:
            return
        lo = self.m.NewIntVar(0, 10**6, "ok_min")
        hi = self.m.NewIntVar(0, 10**6, "ok_max")
        self.m.AddMinEquality(lo, totals)
        self.m.AddMaxEquality(hi, totals)
        spread = self.m.NewIntVar(0, 10**6, "ok_spread")
        self.m.Add(spread == hi - lo)
        self.penalties.append(self.w.ok_fairness * spread)

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

    # ---- ZACHT regel 1: OK in AM en PM op dezelfde dag = dezelfde persoon ----
    def _same_day_pair(self):
        if self.w.same_day_pair <= 0 or not self.pair_skill:
            return
        # groepeer slots van pair_skill per datum
        by_date: dict[str, dict[Period, Slot]] = {}
        for slot in self.slots:
            if slot.required_skill == self.pair_skill:
                by_date.setdefault(slot.date, {})[slot.period] = slot
        for date, periods in by_date.items():
            am, pm = periods.get(Period.AM), periods.get(Period.PM)
            if not (am and pm):
                continue
            # straf als arts s WEL am doet maar NIET pm (en omgekeerd)
            for s in self.staff:
                xa = self.x.get((s.id, am.id))
                xp = self.x.get((s.id, pm.id))
                if xa is None or xp is None:
                    continue
                diff = self.m.NewBoolVar(f"okdiff_{s.id}_{date}")
                # diff >= xa - xp  en  diff >= xp - xa  → diff=1 als ze verschillen
                self.m.Add(diff >= xa - xp)
                self.m.Add(diff >= xp - xa)
                self.penalties.append(self.w.same_day_pair * diff)

    # ---- ZACHT regel 2: niemand Poli in AM EN PM op dezelfde dag -------------
    def _poli_split(self):
        if self.w.poli_split <= 0 or not self.split_skill:
            return
        by_date: dict[str, dict[Period, list]] = {}
        for slot in self.slots:
            if slot.required_skill == self.split_skill:
                by_date.setdefault(slot.date, {}).setdefault(slot.period, []).append(slot)
        for date, periods in by_date.items():
            am_slots = periods.get(Period.AM, [])
            pm_slots = periods.get(Period.PM, [])
            if not (am_slots and pm_slots):
                continue
            for s in self.staff:
                am_vars = [self.x[(s.id, sl.id)] for sl in am_slots if (s.id, sl.id) in self.x]
                pm_vars = [self.x[(s.id, sl.id)] for sl in pm_slots if (s.id, sl.id) in self.x]
                if not (am_vars and pm_vars):
                    continue
                has_am = self.m.NewBoolVar(f"poliam_{s.id}_{date}")
                has_pm = self.m.NewBoolVar(f"polipm_{s.id}_{date}")
                self.m.AddMaxEquality(has_am, am_vars)
                self.m.AddMaxEquality(has_pm, pm_vars)
                both = self.m.NewBoolVar(f"poliboth_{s.id}_{date}")
                # both = has_am AND has_pm
                self.m.AddBoolAnd([has_am, has_pm]).OnlyEnforceIf(both)
                self.m.AddBoolOr([has_am.Not(), has_pm.Not()]).OnlyEnforceIf(both.Not())
                self.penalties.append(self.w.poli_split * both)

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
