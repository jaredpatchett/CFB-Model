"""
Free college football injury report, pulled from Covers.com's public NCAAF
injury page (one page listing every team). Added 9/2026.

Why this source: official conference availability reports (Big Ten, SEC,
etc.) are more authoritative but only cover conference games and each
league formats them differently; Covers aggregates every FBS team on a
single page. Its entries come from team and media reports, not official
filings, so treat statuses as strong hints, not guarantees.

Used as a FILTER and WARNING layer on top of the model, never as a model
input -- the training data has no injury history, so the model can't learn
how many points a player is worth:
  * props: players listed Out/Doubtful can't be official plays;
    Questionable players are flagged.
  * games: key injuries (starting-QB-level especially) are attached to each
    game so the dashboard can warn on the Edge Board / Bet Card.
  * the dashboard gets a full Injury Report page.

Parsing is deliberately structure-agnostic (stdlib html.parser, no new
dependency): it walks the page in order, recognizes team headings by
matching them against the known FBS "School Mascot" names from CFBD, and
reads each table row under the current team as Player / POS / Status. If
Covers changes its layout the fetch returns [] with a warning in the run
log -- it never breaks the rest of the pipeline.
"""
import re
import unicodedata
from datetime import datetime, timezone
from html.parser import HTMLParser

import requests

COVERS_URL = "https://www.covers.com/sport/football/ncaaf/injuries"

# Status words as they appear at the start of Covers' status cell,
# e.g. "Out - Undisclosed ( Thu, Sep 17)".
KNOWN_STATUSES = ["Out For Season", "Out", "IR", "Doubtful", "Questionable", "Probable", "Day-To-Day", "Game-Time Decision"]
BLOCKING_STATUSES = {"Out", "Out For Season", "IR", "Doubtful"}
WARNING_STATUSES = {"Questionable", "Game-Time Decision", "Day-To-Day"}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


class _PageWalker(HTMLParser):
    """Flattens the page into document-order tokens: ('text', str) for text
    outside tables, and ('row', [cell texts]) for each table row."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tokens = []
        self._table_depth = 0
        self._row = None
        self._cell = None
        self._skip = 0  # inside <script>/<style>

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            # Each team section links to the team's page, e.g.
            # /sport/football/ncaaf/teams/main/air-force-falcons -- the slug is
            # the full "school mascot" name, which identifies the team no
            # matter how the page nests its tables.
            href = dict(attrs).get("href") or ""
            m = re.search(r"/ncaaf/teams/main/([a-z0-9-]+)", href)
            if m:
                self.tokens.append(("team", m.group(1).replace("-", " ")))
        if tag in ("script", "style"):
            self._skip += 1
        elif tag == "table":
            self._table_depth += 1
        elif tag == "tr" and self._table_depth:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1
        elif tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(" ".join(" ".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if any(c for c in self._row):
                self.tokens.append(("row", self._row))
            self._row = None
        elif tag == "table" and self._table_depth:
            self._table_depth -= 1

    def handle_data(self, data):
        if self._skip:
            return
        if self._cell is not None:
            self._cell.append(data)
        elif not self._table_depth:
            text = " ".join(data.split())
            if text:
                self.tokens.append(("text", text))


def _parse_status(cell: str):
    """'Out - Undisclosed ( Thu, Sep 17)' -> ('Out', 'Undisclosed', 'Thu, Sep 17')."""
    cell = " ".join((cell or "").split())
    reported = None
    m = re.search(r"\(\s*([^)]*?)\s*\)\s*$", cell)
    if m:
        reported = m.group(1).strip() or None
        cell = cell[:m.start()].strip()
    status_part, _, injury = cell.partition(" - ")
    status = None
    for s in KNOWN_STATUSES:
        if status_part.strip().lower().startswith(s.lower()):
            status = s
            break
    return status or (status_part.strip() or None), (injury.strip() or None), reported


def parse_covers_html(html: str, team_full_names: dict) -> list:
    """team_full_names: normalized 'school mascot' -> CFBD school name.
    Returns a list of injury dicts."""
    walker = _PageWalker()
    walker.feed(html)
    injuries, team = [], None
    for kind, val in walker.tokens:
        if kind in ("text", "team"):
            school = team_full_names.get(_norm(val))
            if school:
                team = school
            continue
        if team is None:
            continue
        cells = [c for c in val if c]
        if not cells:
            continue
        if len(cells) == 1 and team_full_names.get(_norm(cells[0])):
            team = team_full_names[_norm(cells[0])]
            continue
        if len(cells) >= 3 and cells[0].lower() == "player":
            continue  # header row
        if len(cells) >= 3:
            status, injury, reported = _parse_status(cells[2])
            if not status:
                continue
            injuries.append({
                "school": team, "player": cells[0], "pos": cells[1].upper(),
                "status": status, "injury": injury, "reported": reported, "note": None,
            })
        elif len(cells) == 1 and injuries and injuries[-1]["school"] == team and injuries[-1]["note"] is None:
            if "no injuries to report" not in cells[0].lower():
                injuries[-1]["note"] = cells[0]
    return injuries


def fetch_injuries(team_lookup: dict) -> list:
    """team_lookup: export_dashboard_data.build_team_lookup() output (keys are
    normalized 'school' and 'school mascot' forms, values carry 'school' and
    'mascot'). Returns [] on any failure -- callers treat that as 'no injury
    data this run', never as 'nobody is hurt'."""
    full_names = {}
    for rec in team_lookup.values():
        school, mascot = rec.get("school"), rec.get("mascot")
        if school and mascot:
            full_names[_norm(f"{school} {mascot}")] = school
    resp = requests.get(COVERS_URL, timeout=30, headers={
        "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html",
    })
    resp.raise_for_status()
    injuries = parse_covers_html(resp.text, full_names)
    if not injuries:
        title = re.search(r"<title[^>]*>(.*?)</title>", resp.text, re.S | re.I)
        print(f"  [diag] injury page: HTTP {resp.status_code}, {len(resp.text):,} chars, "
              f"title={title.group(1).strip()[:80]!r}, "
              f"team links={len(re.findall(r'/ncaaf/teams/main/', resp.text))}, "
              f"table rows={resp.text.lower().count('<tr')}, "
              f"known team names loaded={len(full_names)}")
    return injuries


def _initial_last(name: str):
    """'Trent Mosley' / 'T. Mosley' / 'Nolan James Jr.' -> ('t', 'mosley')."""
    parts = [p for p in _norm(name).split() if p not in ("jr", "sr", "ii", "iii", "iv", "v")]
    if len(parts) < 2:
        return None
    return parts[0][0], parts[-1]


def build_injury_index(injuries: list) -> dict:
    """(school, first initial, last name) -> injury dict, for matching props
    (full names) against Covers' abbreviated names ('T. Mosley')."""
    idx = {}
    for inj in injuries:
        key = _initial_last(inj["player"])
        if key:
            idx[(inj["school"], key[0], key[1])] = inj
    return idx


def match_player(injury_index: dict, school: str, player_name: str):
    key = _initial_last(player_name)
    if not school or not key:
        return None
    return injury_index.get((school, key[0], key[1]))


def team_injury_summary(injuries: list, school: str) -> list:
    """Compact list of this team's Out/Doubtful/Questionable players, QBs first."""
    rows = [i for i in injuries if i["school"] == school
            and (i["status"] in BLOCKING_STATUSES or i["status"] in WARNING_STATUSES)]
    rows.sort(key=lambda i: (i["pos"] != "QB", i["status"] not in BLOCKING_STATUSES))
    return [{"player": i["player"], "pos": i["pos"], "status": i["status"], "injury": i["injury"]} for i in rows]


def fetched_at() -> str:
    return datetime.now(timezone.utc).isoformat()
