#!/usr/bin/env python3
'''
Typing Trainer - Intelligent CLI Typing Practice Tool
=====================================================

Features
--------
1) Interactive exercises to improve typing speed (WPM) and accuracy with **real-time feedback**.
2) Progress tracking in SQLite (default: ~/.typing_trainer/metrics.db) with per-session history.
3) **Adaptive practice**: detects weak keys and letter pairs (bigrams), then auto-generates focused drills.
4) Post-session analysis: detailed stats (WPM, accuracy, error frequency by key/bigram/word) + tips.
5) Modular design to enable future GUI adaptation. Clear separation of concerns (store/generator/ui/metrics).
6) Works in a text terminal using Python's standard `curses` for cross-platform TUI.

Quick Start
-----------
- Install dependencies: only Python 3.x standard library required.
- Run: `python3 typing_trainer.py`

Command-Line Options
--------------------
- `--duration SECONDS`  : Time-box a session (default: 60).
- `--lines N`           : Number of prompt lines to practice instead of a timer. If set, overrides duration.
- `--mode MODE`         : "auto" (default), "words", "sentences", or "custom".
- `--db PATH`           : Override database path. Default: ~/.typing_trainer/metrics.db
- `--no-store`          : Do not persist results (practice-only).
- `--report`            : Print lifetime stats and recent trends, then exit (no TUI session).
- `--export-csv PATH`   : Export sessions/errors to CSV (PATH_sessions.csv & PATH_errors.csv), then exit.
- `--export-json PATH`  : Export sessions+errors to a single JSON file, then exit.
- `--plot PREFIX`       : Save charts as PREFIX_wpm.png and PREFIX_accuracy.png (raw + 5-SMA), then exit.
- `--help`              : Show help.

Controls (during a session)
---------------------------
- Type the displayed line. Mistakes are shown in red; correct chars are green.
- Use Backspace to correct. Press Enter to submit a line.
- Press ESC to end the session early.
- After ending, a detailed summary is shown; press any key to return.

Design Notes
------------
- SQLite schema: sessions + errors detail tables (auto-created).
- Adaptive generator: weighted sampling favors weak keys/bigrams/words.
- WPM calculation: (chars/5)/minutes; net WPM subtracts an error penalty.
- Modules: DataStore, ExerciseGenerator, SessionEngine, SessionMetrics.
'''

from __future__ import annotations
import argparse
import collections
import curses
import datetime as dt
import json
import random
import sqlite3
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# ------------------------------
# Utility helpers
# ------------------------------

def clamp(n, lo, hi):
    return max(lo, min(hi, n))

def now_ts() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

def human_duration(seconds: float) -> str:
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"

# ------------------------------
# Data Store (SQLite)
# ------------------------------

DEFAULT_DB = str(Path.home() / ".typing_trainer" / "metrics.db")

class DataStore:
    def __init__(self, db_path: str, persist: bool = True):
        self.db_path = db_path
        self.persist = persist
        if self.persist:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(db_path)
            self.conn.execute("PRAGMA journal_mode=WAL")
            self._init_schema()
        else:
            self.conn = None

    def _init_schema(self):
        cur = self.conn.cursor()
        cur.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            duration_sec REAL NOT NULL,
            mode TEXT NOT NULL,
            wpm_gross REAL NOT NULL,
            wpm_net REAL NOT NULL,
            accuracy REAL NOT NULL,
            chars_typed INTEGER NOT NULL,
            chars_correct INTEGER NOT NULL,
            errors_total INTEGER NOT NULL,
            config_json TEXT NOT NULL
        );
        ''')
        cur.execute('''
        CREATE TABLE IF NOT EXISTS errors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            key TEXT,
            bigram TEXT,
            word TEXT,
            count INTEGER NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(id)
        );
        ''')
        self.conn.commit()

    def insert_session(self, metrics: "SessionMetrics", mode: str, duration_sec: float, config: dict) -> int:
        if not self.persist:
            return -1
        cur = self.conn.cursor()
        cur.execute('''
        INSERT INTO sessions (started_at, duration_sec, mode, wpm_gross, wpm_net, accuracy, chars_typed, chars_correct, errors_total, config_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        ''', (
            metrics.started_at,
            duration_sec,
            mode,
            metrics.wpm_gross,
            metrics.wpm_net,
            metrics.accuracy,
            metrics.chars_typed,
            metrics.chars_correct,
            metrics.errors_total,
            json.dumps(config, ensure_ascii=False),
        ))
        sid = cur.lastrowid
        for key, cnt in metrics.error_by_key.items():
            cur.execute('INSERT INTO errors (session_id, key, bigram, word, count) VALUES (?, ?, NULL, NULL, ?);', (sid, key, cnt))
        for bg, cnt in metrics.error_by_bigram.items():
            cur.execute('INSERT INTO errors (session_id, key, bigram, word, count) VALUES (?, NULL, ?, NULL, ?);', (sid, bg, cnt))
        for w, cnt in metrics.error_by_word.items():
            cur.execute('INSERT INTO errors (session_id, key, bigram, word, count) VALUES (?, NULL, NULL, ?, ?);', (sid, w, cnt))
        self.conn.commit()
        return sid

    def lifetime_summary(self) -> dict:
        if not self.persist:
            return {}
        cur = self.conn.cursor()
        summary = {}

        cur.execute('SELECT COUNT(*), AVG(wpm_net), AVG(accuracy) FROM sessions;')
        row = cur.fetchone()
        summary["sessions"] = row[0] or 0
        summary["avg_wpm_net"] = row[1] or 0.0
        summary["avg_accuracy"] = row[2] or 0.0

        cur.execute('SELECT started_at, wpm_net, accuracy, errors_total FROM sessions ORDER BY id DESC LIMIT 10;')
        rows = cur.fetchall()
        summary["recent"] = [{"started_at": r[0], "wpm_net": r[1], "accuracy": r[2], "errors": r[3]} for r in rows]

        def top(table_col):
            cur.execute(f'''
              SELECT {table_col}, SUM(count) as c
              FROM errors WHERE {table_col} IS NOT NULL
              GROUP BY {table_col} ORDER BY c DESC LIMIT 10;
            ''')
            return cur.fetchall()

        summary["top_keys"] = top("key")
        summary["top_bigrams"] = top("bigram")
        summary["top_words"] = top("word")
        return summary

    # NEW: full sessions and errors fetch for exports/dashboards
    def fetch_sessions(self) -> list[dict]:
        if not self.persist:
            return []
        cur = self.conn.cursor()
        cur.execute("""
          SELECT id, started_at, duration_sec, mode, wpm_gross, wpm_net, accuracy,
                 chars_typed, chars_correct, errors_total, config_json
          FROM sessions ORDER BY id ASC;
        """)
        rows = cur.fetchall()
        return [{
            "id": r[0],
            "started_at": r[1],
            "duration_sec": r[2],
            "mode": r[3],
            "wpm_gross": r[4],
            "wpm_net": r[5],
            "accuracy": r[6],
            "chars_typed": r[7],
            "chars_correct": r[8],
            "errors_total": r[9],
            "config_json": r[10],
        } for r in rows]

    def fetch_errors(self) -> list[dict]:
        if not self.persist:
            return []
        cur = self.conn.cursor()
        cur.execute("""
          SELECT session_id, key, bigram, word, count
          FROM errors ORDER BY session_id ASC, id ASC;
        """)
        rows = cur.fetchall()
        return [{
            "session_id": r[0],
            "key": r[1],
            "bigram": r[2],
            "word": r[3],
            "count": r[4],
        } for r in rows]

# ------------------------------
# Metrics
# ------------------------------

@dataclass
class SessionMetrics:
    started_at: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    chars_typed: int = 0
    chars_correct: int = 0
    errors_total: int = 0
    submitted_lines: int = 0
    error_by_key: Dict[str, int] = field(default_factory=lambda: collections.defaultdict(int))
    error_by_bigram: Dict[str, int] = field(default_factory=lambda: collections.defaultdict(int))
    error_by_word: Dict[str, int] = field(default_factory=lambda: collections.defaultdict(int))

    @property
    def elapsed(self) -> float:
        return (self.end_time or time.time()) - self.start_time

    @property
    def accuracy(self) -> float:
        if self.chars_typed == 0:
            return 1.0
        return self.chars_correct / max(1, self.chars_typed)

    @property
    def wpm_gross(self) -> float:
        minutes = max(self.elapsed / 60.0, 1e-6)
        return (self.chars_typed / 5.0) / minutes

    @property
    def wpm_net(self) -> float:
        minutes = max(self.elapsed / 60.0, 1e-6)
        penalty = self.errors_total / minutes
        return max(0.0, self.wpm_gross - penalty)

    def register_keystroke(self, expected: Optional[str], typed: Optional[str]):
        self.chars_typed += 1
        if expected == typed:
            self.chars_correct += 1
        else:
            self.errors_total += 1
            if typed:
                self.error_by_key[typed] += 1
            if expected and typed:
                self.error_by_bigram[(expected + typed).lower()] += 1

    def register_line_result(self, expected_line: str, typed_line: str):
        self.submitted_lines += 1
        exp_words = expected_line.split()
        typ_words = typed_line.split()

        for idx, word in enumerate(exp_words):
            typed = typ_words[idx] if idx < len(typ_words) else ""
            if word != typed:
                self.error_by_word[word] += 1

        for word in typ_words[len(exp_words):]:
            self.error_by_word[word] += 1

# ------------------------------
# Exercise Generator (adaptive)
# ------------------------------

COMMON_WORDS = """
the of and to in is you that it he was for on are as with his they I at be this
have from or one had by word but not what all were we when your can said there
use an each which she do how their if will up other about out many then them
these so some her would make like him into time has look two more write go see
number no way could people my than first water been call who oil its now find
long down day did get come made may part
""".strip().split()

PANGRAMS = [
    "the quick brown fox jumps over the lazy dog",
    "pack my box with five dozen liquor jugs",
    "sphinx of black quartz judge my vow",
    "how vexingly quick daft zebras jump",
]

SENTENCES = [
    "practice makes progress not perfection",
    "type steadily keep shoulders relaxed wrists neutral",
    "accuracy first then speed will follow naturally",
    "focus on smooth rhythm and even keystrokes",
    "short breaks help maintain consistency and form",
]

class ExerciseGenerator:
    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    @staticmethod
    def _weak_items(weak_map: Dict[str, int], k: int = 5) -> List[str]:
        return [w for w, _ in sorted(weak_map.items(), key=lambda x: x[1], reverse=True)[:k]]

    def build_prompt_lines(self, mode: str, weaknesses: Dict[str, Dict[str, int]], lines: int = 10) -> List[str]:
        out = []
        weak_keys = self._weak_items(weaknesses.get("keys", {}), 7)
        weak_bigrams = self._weak_items(weaknesses.get("bigrams", {}), 7)
        weak_words = self._weak_items(weaknesses.get("words", {}), 7)

        def word_weight(w: str) -> float:
            base = 1.0
            for ch in weak_keys:
                if ch and ch in w:
                    base += 1.0
            for bg in weak_bigrams:
                if bg and bg in w.lower():
                    base += 1.0
            if w in weak_words:
                base += 2.0
            return base

        if mode == "sentences":
            pool = SENTENCES + PANGRAMS
            for _ in range(lines): out.append(self.rng.choice(pool))
            return out
        elif mode == "words":
            words = COMMON_WORDS[:]
            weights = [word_weight(w) for w in words]
            for _ in range(lines):
                line = " ".join(self.rng.choices(words, weights=weights, k=8))
                out.append(line)
            return out
        elif mode == "auto":
            words = COMMON_WORDS[:]
            weights = [word_weight(w) for w in words]
            for _ in range(lines):
                if (weak_keys or weak_bigrams or weak_words) and self.rng.random() < 0.7:
                    line = " ".join(self.rng.choices(words, weights=weights, k=8))
                else:
                    line = self.rng.choice(SENTENCES + PANGRAMS)
                out.append(line)
            return out
        else:
            for _ in range(lines):
                out.append(self.rng.choice(SENTENCES + PANGRAMS))
            return out

# ------------------------------
# Session Engine (curses UI)
# ------------------------------

class SessionEngine:
    COLOR_OK = 1
    COLOR_ERR = 2
    COLOR_DIM = 3
    COLOR_INFO = 4

    def __init__(self, stdscr, generator: ExerciseGenerator, metrics: SessionMetrics, config: dict):
        self.stdscr = stdscr
        self.gen = generator
        self.metrics = metrics
        self.config = config
        self.cursor_x = 0
        self.typed = ""
        self.current_line = ""
        self.line_index = 0
        self.finished = False
        self.start_time = time.time()
        self.deadline = self.start_time + config["duration"] if config["duration"] else None
        self.lines_goal = config.get("lines") or None
        self.prompts = self.gen.build_prompt_lines(config["mode"], config["weaknesses"], lines=config["lines"] or 10)

    def init_colors(self):
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(self.COLOR_OK, curses.COLOR_GREEN, -1)
        curses.init_pair(self.COLOR_ERR, curses.COLOR_RED, -1)
        curses.init_pair(self.COLOR_DIM, curses.COLOR_CYAN, -1)
        curses.init_pair(self.COLOR_INFO, curses.COLOR_YELLOW, -1)

    def draw_header(self):
        elapsed = time.time() - self.start_time
        left = max(0, int(self.deadline - time.time())) if self.deadline else None
        info = f"Line {self.line_index+1}/{len(self.prompts)}  |  WPM(net): {self.metrics.wpm_net:.1f}  Acc: {self.metrics.accuracy*100:5.1f}%  Time: {human_duration(elapsed)}"
        if left is not None:
            info += f"  Left: {human_duration(left)}"
        self.stdscr.attron(curses.color_pair(self.COLOR_INFO))
        maxy, maxx = self.stdscr.getmaxyx()
        self.stdscr.addstr(0, 0, info[:maxx-1])
        self.stdscr.attroff(curses.color_pair(self.COLOR_INFO))
        self.stdscr.hline(1, 0, curses.ACS_HLINE, maxx)

    def draw_prompt(self):
        maxy, maxx = self.stdscr.getmaxyx()
        prompt = self.current_line
        self.stdscr.attron(curses.A_BOLD)
        self.stdscr.addstr(3, 0, "Target: "[:maxx-1])
        self.stdscr.attroff(curses.A_BOLD)

        for i, ch in enumerate(prompt):
            y, x = 3, 8 + i
            if x >= maxx-1: break
            if i < len(self.typed):
                exp = ch
                got = self.typed[i]
                if got == exp:
                    self.stdscr.attron(curses.color_pair(self.COLOR_OK))
                    self.stdscr.addch(y, x, ord(exp))
                    self.stdscr.attroff(curses.color_pair(self.COLOR_OK))
                else:
                    self.stdscr.attron(curses.color_pair(self.COLOR_ERR))
                    self.stdscr.addch(y, x, ord(exp))
                    self.stdscr.attroff(curses.color_pair(self.COLOR_ERR))
            else:
                self.stdscr.attron(curses.color_pair(self.COLOR_DIM))
                self.stdscr.addch(y, x, ord(ch))
                self.stdscr.attroff(curses.color_pair(self.COLOR_DIM))

        self.stdscr.attron(curses.A_BOLD)
        self.stdscr.addstr(5, 0, "Typed : "[:maxx-1])
        self.stdscr.attroff(curses.A_BOLD)
        display = self.typed
        max_width = maxx - 8 - 1
        if max_width > 0:
            if len(display) > max_width:
                display = "…" + display[-max_width+1:]
            self.stdscr.addstr(5, 8, display[:max_width])

        self.stdscr.hline(7, 0, curses.ACS_HLINE, maxx)
        self.stdscr.addstr(8, 0, "Enter = submit line | Backspace = correct | ESC = end session"[:maxx-1])

    def next_line(self):
        if self.line_index >= len(self.prompts):
            return False
        self.current_line = self.prompts[self.line_index]
        self.typed = ""
        self.cursor_x = 0
        return True

    def run(self):
        self.init_colors()
        self.stdscr.nodelay(True)
        curses.curs_set(0)
        self.next_line()

        while not self.finished:
            self.stdscr.erase()
            self.draw_header()
            self.draw_prompt()
            self.stdscr.refresh()

            if self.deadline and time.time() >= self.deadline:
                self.finished = True
                break
            if self.lines_goal and self.metrics.submitted_lines >= self.lines_goal:
                self.finished = True
                break

            try:
                ch = self.stdscr.getch()
            except curses.error:
                ch = -1

            if ch == -1:
                time.sleep(0.01)
                continue

            if ch == 27:
                self.finished = True
                break
            elif ch in (curses.KEY_BACKSPACE, 127, 8):
                if len(self.typed) > 0:
                    self.typed = self.typed[:-1]
                    self.cursor_x = max(0, self.cursor_x - 1)
                continue
            elif ch in (10, 13):
                self.metrics.register_line_result(self.current_line, self.typed)
                self.line_index += 1
                if not self.next_line():
                    self.finished = True
                continue
            elif 0 <= ch <= 255:
                c = chr(ch)
                exp = self.current_line[self.cursor_x] if self.cursor_x < len(self.current_line) else None
                self.metrics.register_keystroke(exp, c)
                self.typed += c
                self.cursor_x += 1

        self.metrics.end_time = time.time()

# ------------------------------
# Summary rendering (curses)
# ------------------------------

def render_summary(stdscr, metrics: SessionMetrics, weaknesses: Dict[str, Dict[str, int]]):
    stdscr.erase()
    curses.curs_set(0)

    def safe_add_line(y: int, text: str, attr: int = 0) -> int:
        maxy, maxx = stdscr.getmaxyx()
        if maxx <= 1 or y >= maxy - 1:
            return y
        width = max(1, maxx - 1)
        for wrapped in textwrap.wrap(text, width=width):
            if y >= maxy - 1:
                return y
            try:
                stdscr.addstr(y, 0, wrapped[:width], attr)
            except curses.error:
                pass
            y += 1
        return y

    def maybe_sep(y: int) -> int:
        maxy, maxx = stdscr.getmaxyx()
        if y < maxy - 1:
            try:
                stdscr.hline(y, 0, curses.ACS_HLINE, maxx)
            except curses.error:
                pass
            y += 1
        return y

    def topn(d: Dict[str, int], n=5):
        return sorted(d.items(), key=lambda x: x[1], reverse=True)[:n]

    y = 0
    y = safe_add_line(y, "Session Summary", curses.A_BOLD | curses.A_UNDERLINE); y += 1
    y = safe_add_line(y, f"WPM (gross): {metrics.wpm_gross:.1f}")
    y = safe_add_line(y, f"WPM (net)  : {metrics.wpm_net:.1f}")
    y = safe_add_line(y, f"Accuracy   : {metrics.accuracy*100:.1f}%")
    y = safe_add_line(y, f"Typed chars: {metrics.chars_typed} | Correct: {metrics.chars_correct} | Errors: {metrics.errors_total}")
    y = safe_add_line(y, f"Lines done : {metrics.submitted_lines}")
    y = maybe_sep(y)

    maxy, _ = stdscr.getmaxyx()
    def space_left():
        return max(0, (maxy - 2) - y)

    y = safe_add_line(y, "Top weak KEYS:", curses.A_BOLD)
    keys = topn(metrics.error_by_key, 10)
    for k, c in keys:
        if space_left() == 0: break
        y = safe_add_line(y, f"  {repr(k)} : {c}")
    if len(keys) > space_left() and space_left() > 0:
        y = safe_add_line(y, "  …")

    if space_left() > 0:
        y += 1
        y = safe_add_line(y, "Top weak BIGRAMS:", curses.A_BOLD)
        bgs = topn(metrics.error_by_bigram, 10)
        for bg, c in bgs:
            if space_left() == 0: break
            y = safe_add_line(y, f"  {bg} : {c}")
        if len(bgs) > space_left() and space_left() > 0:
            y = safe_add_line(y, "  …")

    if space_left() > 0:
        y += 1
        y = safe_add_line(y, "Top weak WORDS:", curses.A_BOLD)
        words = topn(metrics.error_by_word, 10)
        for w, c in words:
            if space_left() == 0: break
            y = safe_add_line(y, f"  {w} : {c}")
        if len(words) > space_left() and space_left() > 0:
            y = safe_add_line(y, "  …")

    if space_left() > 0:
        y += 1
        y = maybe_sep(y)
        y = safe_add_line(y, "Suggestions:", curses.A_BOLD)
        for t in suggest_improvements(metrics):
            if space_left() == 0: break
            y = safe_add_line(y, f"- {t}")

    y = safe_add_line(y, "Press any key to continue...")
    stdscr.refresh()
    stdscr.getch()

def suggest_improvements(metrics: SessionMetrics) -> List[str]:
    tips = []
    if metrics.accuracy < 0.9:
        tips.append("Prioritize accuracy over speed: slow down slightly and aim for 95%+ before pushing WPM.")
    if metrics.wpm_net < 35:
        tips.append("Practice short bursts (1–2 minutes) focusing on smooth, consistent keystrokes to build rhythm.")
    for k, cnt in sorted(metrics.error_by_key.items(), key=lambda x: x[1], reverse=True)[:3]:
        if k.strip():
            tips.append(f"Spend 2 minutes on focused drills for key '{k}'. Keep fingers light and return to home row.")
    for bg, cnt in sorted(metrics.error_by_bigram.items(), key=lambda x: x[1], reverse=True)[:2]:
        if bg.strip():
            tips.append(f"Type 10 lines with words containing '{bg}' (e.g., make a quick tongue-twister).")
    if metrics.errors_total == 0 and metrics.chars_typed > 0:
        tips.append("Excellent accuracy! Start nudging speed up by 5–10% while keeping form.")
    return tips

# ------------------------------
# Reporting (CLI non-TUI)
# ------------------------------

def print_report(store: DataStore):
    summ = store.lifetime_summary()
    if not summ:
        print("No data available (persistence disabled or no sessions yet).")
        return
    print("\n=== Lifetime Summary ===")
    print(f"Total sessions     : {summ['sessions']}")
    print(f"Avg WPM (net)      : {summ['avg_wpm_net']:.1f}")
    print(f"Avg Accuracy       : {summ['avg_accuracy']*100:.1f}%")

    print("\n=== Recent (latest 10) ===")
    for r in summ["recent"]:
        print(f"{r['started_at']} | WPM(net): {r['wpm_net']:.1f} | Acc: {r['accuracy']*100:5.1f}% | Errors: {r['errors']}")

    def show_top(label, items):
        print(f"\nTop {label}:")
        for item, c in items:
            print(f"  {item!r:<12} -> {c}")
    show_top("keys", summ["top_keys"])
    show_top("bigrams", summ["top_bigrams"])
    show_top("words", summ["top_words"])

# ------------------------------
# Weakness extraction (from history)
# ------------------------------

def historical_weaknesses(store: DataStore) -> Dict[str, Dict[str, int]]:
    if not store.persist:
        return {"keys": {}, "bigrams": {}, "words": {}}
    cur = store.conn.cursor()
    def agg(col):
        cur.execute(f"SELECT {col}, SUM(count) FROM errors WHERE {col} IS NOT NULL GROUP BY {col};")
        return {row[0]: int(row[1]) for row in cur.fetchall()}
    return {"keys": agg("key"), "bigrams": agg("bigram"), "words": agg("word")}

# ------------------------------
# Exports & Charts
# ------------------------------

def export_csv(store: DataStore, path: str):
    import csv
    sessions = store.fetch_sessions()
    errors = store.fetch_errors()
    # sessions CSV
    sessions_path = f"{path}_sessions.csv"
    if sessions:
        with open(sessions_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = list(sessions[0].keys())
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in sessions:
                w.writerow(row)
        print(f"CSV exported -> {sessions_path}")
    else:
        print("No sessions to export.")
    # errors CSV
    errors_path = f"{path}_errors.csv"
    if errors:
        with open(errors_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = list(errors[0].keys())
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in errors:
                w.writerow(row)
        print(f"CSV exported -> {errors_path}")
    else:
        print("No errors found to write CSV.")

def export_json(store: DataStore, path: str):
    data = {"sessions": store.fetch_sessions(), "errors": store.fetch_errors()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"JSON exported -> {path}")

def _sma(values: list[float], window: int = 5) -> list[Optional[float]]:
    if window <= 1:
        return values[:]
    out, s, q = [], 0.0, []
    for v in values:
        q.append(v); s += v
        if len(q) > window: s -= q.pop(0)
        out.append(s/len(q) if len(q)==window else None)
    return out

def plot_charts(store: DataStore, prefix: str):
    import matplotlib.pyplot as plt
    data = store.fetch_sessions()
    if not data:
        print("No sessions to plot.")
        return
    data = sorted(data, key=lambda d: d["id"])
    xs = [d["id"] for d in data]
    wpm = [d["wpm_net"] for d in data]
    acc = [d["accuracy"]*100.0 for d in data]

    # WPM chart
    plt.figure()
    plt.plot(xs, wpm, marker="o", label="Net WPM")
    sma_wpm = _sma(wpm, 5)
    xs_sma = [x for x, y in zip(xs, sma_wpm) if y is not None]
    y_sma = [y for y in sma_wpm if y is not None]
    if xs_sma: plt.plot(xs_sma, y_sma, linestyle="--", label="Net WPM (5-SMA)")
    plt.title("Net WPM over sessions")
    plt.xlabel("Session #"); plt.ylabel("Net WPM"); plt.legend()
    wpm_path = f"{prefix}_wpm.png"; plt.savefig(wpm_path, bbox_inches="tight"); plt.close()
    print(f"Saved {wpm_path}")

    # Accuracy chart
    plt.figure()
    plt.plot(xs, acc, marker="o", label="Accuracy (%)")
    sma_acc = _sma(acc, 5)
    xs_sma = [x for x, y in zip(xs, sma_acc) if y is not None]
    y_sma = [y for y in sma_acc if y is not None]
    if xs_sma: plt.plot(xs_sma, y_sma, linestyle="--", label="Accuracy (5-SMA)")
    plt.title("Accuracy (%) over sessions")
    plt.xlabel("Session #"); plt.ylabel("Accuracy (%)"); plt.legend()
    acc_path = f"{prefix}_accuracy.png"; plt.savefig(acc_path, bbox_inches="tight"); plt.close()
    print(f"Saved {acc_path}")

# ------------------------------
# Run a session (wrapping curses)
# ------------------------------

def run_session(config: dict, store: DataStore):
    weaknesses = historical_weaknesses(store)
    gen = ExerciseGenerator()
    metrics = SessionMetrics()
    config = dict(config)
    config["weaknesses"] = weaknesses

    def _session(stdscr):
        engine = SessionEngine(stdscr, gen, metrics, config)
        engine.run()
        render_summary(stdscr, metrics, weaknesses)

    curses.wrapper(_session)
    sid = -1
    if store.persist:
        sid = store.insert_session(metrics, mode=config["mode"], duration_sec=metrics.elapsed, config=config)
    return metrics, sid

# ------------------------------
# Argparse / Main
# ------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="CLI intelligent typing trainer")
    p.add_argument("--duration", type=int, default=60, help="Session duration in seconds (ignored if --lines is set)")
    p.add_argument("--lines", type=int, default=None, help="Number of lines to practice (overrides duration)")
    p.add_argument("--mode", type=str, default="auto", choices=["auto", "words", "sentences", "custom"], help="Exercise mode")
    p.add_argument("--db", type=str, default=DEFAULT_DB, help="SQLite database path")
    p.add_argument("--no-store", action="store_true", help="Do not store results")
    p.add_argument("--report", action="store_true", help="Print lifetime summary and exit")
    p.add_argument("--export-csv", type=str, default=None, help="Export to CSV base path (writes *_sessions.csv and *_errors.csv), then exit")
    p.add_argument("--export-json", type=str, default=None, help="Export sessions+errors to JSON at path, then exit")
    p.add_argument("--plot", type=str, default=None, help="Save charts to files with this path prefix (e.g., /tmp/typing), then exit")
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    persist = not args.no_store
    store = DataStore(args.db, persist=persist)

    if args.export_csv:
        export_csv(store, args.export_csv); return 0
    if args.export_json:
        export_json(store, args.export_json); return 0
    if args.plot:
        plot_charts(store, args.plot); return 0
    if args.report:
        print_report(store); return 0

    config = {
        "duration": None if args.lines else clamp(args.duration, 10, 3600),
        "lines": clamp(args.lines, 1, 100) if args.lines else None,
        "mode": args.mode,
    }

    try:
        metrics, sid = run_session(config, store)
    except KeyboardInterrupt:
        print("\nSession cancelled."); return 1

    print("\n=== Session Results ===")
    print(f"Started at     : {metrics.started_at}")
    print(f"Elapsed        : {human_duration(metrics.elapsed)}")
    print(f"WPM (gross)    : {metrics.wpm_gross:.1f}")
    print(f"WPM (net)      : {metrics.wpm_net:.1f}")
    print(f"Accuracy       : {metrics.accuracy*100:.1f}%")
    print(f"Typed chars    : {metrics.chars_typed}  | Correct: {metrics.chars_correct}  | Errors: {metrics.errors_total}")
    print(f"Lines submitted: {metrics.submitted_lines}")
    if persist and sid != -1:
        print(f"Saved to DB     : {args.db} (session id {sid})")
    else:
        print("Results not persisted.")

    print("\nTips:")
    for t in suggest_improvements(metrics):
        print(f"- {t}")

    print("\nNext steps:")
    print("  * Run `python3 typing_trainer.py --report` to see your lifetime stats and recent trends.")
    print("  * Try a focused session: `python3 typing_trainer.py --mode words --lines 20`")
    print("  * Or a quick 2-min drill: `python3 typing_trainer.py --duration 120`")
    return 0

if __name__ == "__main__":
    sys.exit(main())
