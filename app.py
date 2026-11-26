"""SpectreVision demo site: login (SQLite) + two feature tabs with the processed videos.

Run:   python app.py      then open http://127.0.0.1:5000
"""
import os
import sqlite3
from functools import wraps

from flask import Flask, render_template_string, request, redirect, url_for, session, send_from_directory, abort
from werkzeug.security import generate_password_hash, check_password_hash

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "users.db")
VIDEOS = os.path.join(HERE, "videos")

app = Flask(__name__)
_key_file = os.path.join(HERE, ".secret_key")
if not os.path.exists(_key_file):
    with open(_key_file, "w") as fh:
        fh.write(os.urandom(24).hex())
app.secret_key = open(_key_file).read().strip()

DEFAULT_USER, DEFAULT_PASS = "spectrevision", "SV-demo-2026"

FEATURES = {
    "loading": {
        "title": "Loading / Unloading",
        "video": "loading_demo.mp4",
        "desc": "A custom YOLO model detects boxes and people carrying boxes. Each is tracked across "
                "frames, and a box is counted as LOADED when it crosses the line at the back of the truck "
                "while a person carrying a box is present.",
        "facts": [("Model", "Custom YOLO (box, person carrying box)"), ("Counted in clip", "13 boxes loaded"),
                  ("Clip", "4 min, packing area camera")],
    },
    "helmet": {
        "title": "Helmet Detection",
        "video": "helmet_demo.mp4",
        "desc": "Two stages: YOLOv8 first finds and tracks every person, then a custom-trained helmet model "
                "checks each person's head and marks them Helmet (green) or No-Helmet (red).",
        "facts": [("Models", "YOLOv8s (people) + custom YOLO11s (Helmet / No-Helmet)"),
                  ("Training data", "176 labelled images, 243 boxes"), ("Clip", "5 min, foundry yard camera")],
    },
}


# ── database ──────────────────────────────────────────────────────────────────
def db():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with db() as con:
        con.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL)")
        if not con.execute("SELECT 1 FROM users").fetchone():
            con.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)",
                        (DEFAULT_USER, generate_password_hash(DEFAULT_PASS)))


def login_required(f):
    @wraps(f)
    def wrapper(*a, **kw):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*a, **kw)
    return wrapper


# ── pages ─────────────────────────────────────────────────────────────────────
BASE_CSS = """
<style>
:root{--bg:#0d1117;--card:#161b22;--line:#30363d;--text:#e6edf3;--muted:#8b949e;--accent:#2f81f7}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Segoe UI,Arial,sans-serif}
header{display:flex;align-items:center;justify-content:space-between;padding:14px 28px;border-bottom:1px solid var(--line)}
.brand{font-size:22px;font-weight:700;letter-spacing:1px}.brand span{color:var(--accent)}
nav a{color:var(--muted);text-decoration:none;margin-left:22px;padding:8px 0;font-weight:600}
nav a.active,nav a:hover{color:var(--text);border-bottom:2px solid var(--accent)}
main{max-width:1100px;margin:28px auto;padding:0 20px}
video{width:100%;border-radius:10px;border:1px solid var(--line);background:#000}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:20px;margin-top:18px}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-top:12px}
.fact{background:#0d1117;border:1px solid var(--line);border-radius:8px;padding:12px}.fact b{display:block;color:var(--muted);font-size:12px;text-transform:uppercase}
.login{max-width:360px;margin:12vh auto;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:30px}
input{width:100%;padding:11px;margin:8px 0 14px;background:#0d1117;border:1px solid var(--line);border-radius:6px;color:var(--text)}
button{width:100%;padding:11px;background:var(--accent);border:0;border-radius:6px;color:#fff;font-weight:700;cursor:pointer}
.err{color:#f85149;margin-bottom:8px}
</style>"""

LOGIN_HTML = BASE_CSS + """
<title>SpectreVision - Login</title>
<div class="login">
  <div class="brand" style="text-align:center;margin-bottom:20px">SPECTRE<span>VISION</span></div>
  {% if error %}<div class="err">{{ error }}</div>{% endif %}
  <form method="post">
    <label>Username</label><input name="username" autofocus required>
    <label>Password</label><input name="password" type="password" required>
    <button type="submit">Sign in</button>
  </form>
</div>"""

PAGE_HTML = BASE_CSS + """
<title>SpectreVision - {{ f.title }}</title>
<header>
  <div class="brand">SPECTRE<span>VISION</span></div>
  <nav>
    {% for key, item in features.items() %}
      <a href="{{ url_for('feature', name=key) }}" class="{{ 'active' if key == current else '' }}">{{ item.title }}</a>
    {% endfor %}
    <a href="{{ url_for('logout') }}">Logout</a>
  </nav>
</header>
<main>
  <h2>{{ f.title }}</h2>
  <video controls preload="metadata" src="{{ url_for('video', filename=f.video) }}"></video>
  <div class="card">
    <p>{{ f.desc }}</p>
    <div class="facts">{% for k, v in f.facts %}<div class="fact"><b>{{ k }}</b>{{ v }}</div>{% endfor %}</div>
  </div>
</main>"""


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        row = db().execute("SELECT * FROM users WHERE username = ?", (request.form.get("username", ""),)).fetchone()
        if row and check_password_hash(row["password_hash"], request.form.get("password", "")):
            session["user"] = row["username"]
            return redirect(url_for("feature", name="loading"))
        error = "Wrong username or password."
    return render_template_string(LOGIN_HTML, error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def home():
    return redirect(url_for("feature", name="loading"))


@app.route("/feature/<name>")
@login_required
def feature(name):
    if name not in FEATURES:
        abort(404)
    return render_template_string(PAGE_HTML, f=FEATURES[name], features=FEATURES, current=name)


@app.route("/videos/<path:filename>")
@login_required
def video(filename):
    return send_from_directory(VIDEOS, filename, conditional=True)   # conditional = seeking works


if __name__ == "__main__":
    init_db()
    print(f"\nSpectreVision demo running at http://127.0.0.1:5000   (login: {DEFAULT_USER} / {DEFAULT_PASS})\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
