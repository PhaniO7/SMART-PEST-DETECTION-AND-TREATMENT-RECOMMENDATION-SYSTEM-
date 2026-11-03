"""
================================================================================
FILE: database.py
PURPOSE:
    Manages the local SQLite database (`pest_system.db`) using Python's built-in
    `sqlite3` module and hashes passwords using Python's standard library
    `hashlib` (`pbkdf2_hmac` with SHA-256).

    Stores 4 tables:
      1. `users`              — Registered farmers/students and admins (`role` =
                                "user" or "admin").
      2. `scans`              — Every leaf scan history record, including the
                                measured `dsa_speed_ms` from `time.perf_counter()`.
      3. `feedback`           — Complaints/feedback submitted by users (with name,
                                email, date/time, status) and admin response
                                messages visible in the user's account.
      4. `evaluation_metrics` — Real test-set accuracy, macro F1, YOLO mAP50, and
                                evaluation timestamp saved by `evaluate.py` and
                                displayed dynamically on the website.

HOW IT CONNECTS TO OTHER FILES:
    - Uses `DATABASE_PATH`, `ADMIN_EMAILS`, `PASSWORD_HASH_SALT`, and `DEMO_MODE`
      from `config.py`.
    - Called by `evaluate.py` to store real test-set evaluation metrics.
    - Called by `app.py` for login/registration, saving scans, reading real
      detection accuracy and measured DSA pipeline speed, and managing user
      complaints and admin responses.
    - Tested by `test_all.py`.
================================================================================
"""

# Import standard library modules for SQLite, password hashing, and timestamps
from datetime import datetime
import hashlib
import hmac
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

# Import central project configuration
import config


# ==============================================================================
# 1. CONNECTION & SCHEMA INITIALIZATION (# [DATABASE])
# ==============================================================================

# # [DATABASE]
def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Opens a SQLite connection to `db_path` (defaults to `config.DATABASE_PATH`)
    with `sqlite3.Row` factory so columns can be accessed by name.
    """
    target_path = db_path if db_path is not None else config.DATABASE_PATH
    conn = sqlite3.connect(target_path)
    conn.row_factory = sqlite3.Row
    return conn


# # [DATABASE] # [LOGIN] # [ADMIN] # [FEEDBACK]
def init_db(db_path: Optional[str] = None) -> None:
    """
    Creates the `users`, `scans`, `feedback`, and `evaluation_metrics` tables if
    they do not exist yet.
    If `config.DEMO_MODE` is True, also seeds safe demo accounts. When
    `config.DEMO_MODE` is False (the default for showing or sharing the project),
    no demo accounts are seeded.
    """
    conn = get_connection(db_path)
    cur = conn.cursor()

    # 1. Users table (role is 'user' or 'admin')
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            created_at TEXT NOT NULL
        )
        """
    )

    # 2. Scans table (stores each pest detection result + measured dsa_speed_ms)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            user_name TEXT NOT NULL,
            user_email TEXT NOT NULL,
            crop TEXT NOT NULL,
            pest_name TEXT NOT NULL,
            scientific_name TEXT NOT NULL,
            cnn_confidence REAL NOT NULL,
            yolo_confidence REAL NOT NULL,
            affected_area_pct REAL NOT NULL,
            severity TEXT NOT NULL,
            top_treatment TEXT NOT NULL,
            dsa_speed_ms REAL NOT NULL,
            image_path TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )

    # 3. Feedback / Complaints table (tracks who complained, date/time, and admin reply)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            user_name TEXT NOT NULL,
            user_email TEXT NOT NULL,
            scan_id INTEGER,
            crop TEXT DEFAULT '',
            pest_name TEXT DEFAULT '',
            subject TEXT NOT NULL,
            message TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Pending',
            admin_reply TEXT DEFAULT '',
            replied_by TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            replied_at TEXT DEFAULT '',
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )

    # 4. Real Model Evaluation Metrics table (written by evaluate.py, read by app.py)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS evaluation_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cnn_test_accuracy REAL NOT NULL,
            cnn_macro_f1 REAL NOT NULL,
            yolo_map50 REAL,
            evaluated_at TEXT NOT NULL
        )
        """
    )

    conn.commit()
    conn.close()

    # Seed demo accounts ONLY if config.DEMO_MODE is explicitly set to True
    if config.DEMO_MODE:
        _seed_demo_accounts(db_path)


# ==============================================================================
# 2. PASSWORD HASHING & USER AUTHENTICATION (# [LOGIN] # [ADMIN])
# ==============================================================================

# # [LOGIN]
def hash_password(plain_password: str) -> str:
    """
    Hashes `plain_password` using Python's built-in `hashlib.pbkdf2_hmac`
    (SHA-256) with `config.PASSWORD_HASH_SALT` and `config.PASSWORD_HASH_ITERATIONS`.
    Returns a hexadecimal digest string.
    """
    salt_bytes = config.PASSWORD_HASH_SALT.encode("utf-8")
    pw_bytes = plain_password.encode("utf-8")
    derived = hashlib.pbkdf2_hmac(
        "sha256", pw_bytes, salt_bytes, config.PASSWORD_HASH_ITERATIONS
    )
    return derived.hex()


# # [LOGIN]
def verify_password(plain_password: str, stored_hash: str) -> bool:
    """
    Verifies `plain_password` against `stored_hash` using constant-time comparison.
    """
    candidate_hash = hash_password(plain_password)
    return hmac.compare_digest(candidate_hash, stored_hash)


# # [LOGIN] # [ADMIN]
def _seed_demo_accounts(db_path: Optional[str] = None) -> None:
    """
    Creates demo admin and farmer accounts ONLY when `config.DEMO_MODE = True`.
    """
    register_user(
        name="Demo Admin",
        email=config.DEMO_ADMIN_EMAIL,
        password=config.DEMO_DEFAULT_PASSWORD,
        role_override="admin",
        db_path=db_path,
    )
    register_user(
        name="Demo Farmer",
        email=config.DEMO_USER_EMAIL,
        password=config.DEMO_DEFAULT_PASSWORD,
        role_override="user",
        db_path=db_path,
    )


# # [LOGIN] # [ADMIN]
def register_user(
    name: str,
    email: str,
    password: str,
    role_override: Optional[str] = None,
    db_path: Optional[str] = None,
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Registers a new user account in SQLite.
    If the normalized email is in `config.ADMIN_EMAILS` (or `role_override` is
    provided), assigns the `'admin'` role; otherwise assigns `'user'`.
    Returns `(success, message, user_dict_or_None)`.
    """
    clean_name = name.strip()
    clean_email = email.strip().lower()

    if not clean_name or not clean_email or not password:
        return False, "Name, email, and password are all required.", None

    if len(password) < 4:
        return False, "Password must be at least 4 characters long.", None

    role = (
        role_override
        if role_override in ("user", "admin")
        else ("admin" if clean_email in config.ADMIN_EMAILS else "user")
    )
    pw_hash = hash_password(password)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = get_connection(db_path)
    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO users (name, email, password_hash, role, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (clean_name, clean_email, pw_hash, role, now_str),
        )
        conn.commit()
        user_id = cur.lastrowid
        user_dict = {
            "id": user_id,
            "name": clean_name,
            "email": clean_email,
            "role": role,
            "created_at": now_str,
        }
        return True, f"Account created successfully ({role}).", user_dict
    except sqlite3.IntegrityError:
        return False, "An account with this email already exists.", None
    finally:
        conn.close()


# # [LOGIN]
def authenticate_user(
    email: str, password: str, db_path: Optional[str] = None
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Authenticates a user by email and password against the SQLite `users` table.
    Returns `(success, message, user_dict_or_None)`.
    """
    clean_email = email.strip().lower()
    if not clean_email or not password:
        return False, "Please enter both email and password.", None

    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE email = ?", (clean_email,))
    row = cur.fetchone()
    conn.close()

    if row is None:
        return False, "No account found with that email. Please register first.", None

    if not verify_password(password, row["password_hash"]):
        return False, "Incorrect password. Please try again.", None

    return (
        True,
        f"Welcome back, {row['name']}!",
        {
            "id": row["id"],
            "name": row["name"],
            "email": row["email"],
            "role": row["role"],
            "created_at": row["created_at"],
        },
    )


# ==============================================================================
# 3. SCAN HISTORY & MEASURED DSA PIPELINE SPEED (# [DATABASE])
# ==============================================================================

# # [DATABASE]
def save_scan(
    user_id: int,
    user_name: str,
    user_email: str,
    crop: str,
    pest_name: str,
    scientific_name: str,
    cnn_confidence: float,
    yolo_confidence: float,
    affected_area_pct: float,
    severity: str,
    top_treatment: str,
    dsa_speed_ms: float,
    image_path: str = "",
    db_path: Optional[str] = None,
) -> int:
    """
    Saves a completed leaf scan (including its real measured `dsa_speed_ms`)
    to the SQLite `scans` table and returns the new scan ID.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO scans (
            user_id, user_name, user_email, crop, pest_name, scientific_name,
            cnn_confidence, yolo_confidence, affected_area_pct, severity,
            top_treatment, dsa_speed_ms, image_path, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(user_id),
            user_name,
            user_email,
            crop,
            pest_name,
            scientific_name,
            round(float(cnn_confidence), 4),
            round(float(yolo_confidence), 4),
            round(float(affected_area_pct), 2),
            severity,
            top_treatment,
            round(float(dsa_speed_ms), 3),
            image_path,
            now_str,
        ),
    )
    conn.commit()
    scan_id = int(cur.lastrowid)
    conn.close()
    return scan_id


# # [DATABASE]
def get_scans(
    user_id: Optional[int] = None, db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Returns scan history records ordered newest first.
    If `user_id` is given, returns only that user's scans; otherwise returns all
    scans (used by the Admin view).
    """
    conn = get_connection(db_path)
    cur = conn.cursor()
    if user_id is not None:
        cur.execute(
            "SELECT * FROM scans WHERE user_id = ? ORDER BY id DESC", (int(user_id),)
        )
    else:
        cur.execute("SELECT * FROM scans ORDER BY id DESC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


# # [DATABASE]
def get_dsa_speed_stats(
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Computes the real measured DSA pipeline speed from saved scans in SQLite.
    Returns:
      - `has_scans`: True if at least 1 scan exists in SQLite
      - `last_speed_ms`: measured speed (ms) of the most recent scan (or None)
      - `avg_speed_ms`: average speed (ms) across all saved scans (or None)
      - `display_text`: formatted string or `"No scans yet"` when no scans exist
    """
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT dsa_speed_ms FROM scans ORDER BY id DESC"
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return {
            "has_scans": False,
            "last_speed_ms": None,
            "avg_speed_ms": None,
            "total_scans": 0,
            "display_text": "No scans yet",
        }

    speeds = [float(r["dsa_speed_ms"]) for r in rows]
    last_ms = speeds[0]
    avg_ms = sum(speeds) / len(speeds)
    return {
        "has_scans": True,
        "last_speed_ms": round(last_ms, 2),
        "avg_speed_ms": round(avg_ms, 2),
        "total_scans": len(speeds),
        "display_text": f"{last_ms:.2f} ms (avg {avg_ms:.2f} ms)",
    }


# ==============================================================================
# 4. REAL MODEL EVALUATION METRICS (# [DATABASE] # [CNN] # [YOLO])
# ==============================================================================

# # [DATABASE] # [CNN] # [YOLO]
def save_evaluation_metrics(
    cnn_test_accuracy: float,
    cnn_macro_f1: float,
    yolo_map50: Optional[float] = None,
    db_path: Optional[str] = None,
) -> int:
    """
    Called by `evaluate.py` after running the trained CNN and YOLO models on the
    real IP102 test split. Saves:
      - `cnn_test_accuracy` (percentage 0..100)
      - `cnn_macro_f1` (percentage 0..100)
      - `yolo_map50` (percentage 0..100 or None)
      - `evaluated_at` (current timestamp)
    into the SQLite `evaluation_metrics` table.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO evaluation_metrics (
            cnn_test_accuracy, cnn_macro_f1, yolo_map50, evaluated_at
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            round(float(cnn_test_accuracy), 2),
            round(float(cnn_macro_f1), 2),
            round(float(yolo_map50), 2) if yolo_map50 is not None else None,
            now_str,
        ),
    )
    conn.commit()
    metric_id = int(cur.lastrowid)
    conn.close()
    return metric_id


# # [DATABASE] # [CNN] # [YOLO]
def get_latest_evaluation_metrics(
    db_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Reads the most recent evaluation record saved by `evaluate.py` from SQLite.
    Returns `None` if `evaluate.py` has not been run yet (so the website shows
    `"Not evaluated yet"` instead of a hardcoded number).
    """
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM evaluation_metrics ORDER BY id DESC LIMIT 1"
    )
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    return dict(row)


# ==============================================================================
# 5. USER FEEDBACK / COMPLAINTS & ADMIN RESPONSES (# [FEEDBACK] # [ADMIN])
# ==============================================================================

# # [FEEDBACK]
def submit_feedback(
    user_id: int,
    user_name: str,
    user_email: str,
    subject: str,
    message: str,
    scan_id: Optional[int] = None,
    crop: str = "",
    pest_name: str = "",
    db_path: Optional[str] = None,
) -> Tuple[bool, str, Optional[int]]:
    """
    Submits a user complaint or feedback ticket with timestamp, user name, and
    email so the admin can view who made the complaint and reply to it.
    """
    clean_subject = subject.strip()
    clean_message = message.strip()
    if not clean_subject or not clean_message:
        return False, "Both subject and message are required.", None

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO feedback (
            user_id, user_name, user_email, scan_id, crop, pest_name,
            subject, message, status, admin_reply, replied_by, created_at, replied_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'Pending', '', '', ?, '')
        """,
        (
            int(user_id),
            user_name.strip(),
            user_email.strip().lower(),
            scan_id,
            crop.strip(),
            pest_name.strip(),
            clean_subject,
            clean_message,
            now_str,
        ),
    )
    conn.commit()
    fb_id = int(cur.lastrowid)
    conn.close()
    return True, f"Complaint #{fb_id} submitted successfully.", fb_id


# # [FEEDBACK] # [ADMIN]
def get_feedback_list(
    user_id: Optional[int] = None, db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Returns feedback/complaint records ordered newest first.
    If `user_id` is provided, returns only that user's complaints (for tracking
    in their account); if `user_id` is None, returns all complaints for the Admin.
    """
    conn = get_connection(db_path)
    cur = conn.cursor()
    if user_id is not None:
        cur.execute(
            "SELECT * FROM feedback WHERE user_id = ? ORDER BY id DESC",
            (int(user_id),),
        )
    else:
        cur.execute("SELECT * FROM feedback ORDER BY id DESC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


# # [FEEDBACK] # [ADMIN]
def reply_to_feedback(
    feedback_id: int,
    admin_reply: str,
    admin_name: str,
    new_status: str = "Resolved",
    db_path: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Saves an admin's response message and updates the complaint status
    (`"Resolved"` or `"Pending"`). The user sees this reply when they log in.
    """
    clean_reply = admin_reply.strip()
    if not clean_reply:
        return False, "Admin reply message cannot be empty."

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE feedback
        SET admin_reply = ?, replied_by = ?, status = ?, replied_at = ?
        WHERE id = ?
        """,
        (clean_reply, admin_name.strip(), new_status, now_str, int(feedback_id)),
    )
    conn.commit()
    updated = cur.rowcount
    conn.close()

    if updated == 0:
        return False, f"Complaint #{feedback_id} was not found."
    return True, f"Response sent for Complaint #{feedback_id}."


# # [ADMIN] # [DATABASE]
def get_all_users(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Returns all registered users (without password hashes) for the Admin page.
    """
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, name, email, role, created_at FROM users ORDER BY id DESC"
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows
