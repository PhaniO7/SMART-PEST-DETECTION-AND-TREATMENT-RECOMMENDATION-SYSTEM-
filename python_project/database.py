"""
================================================================================
FILE: database.py
PURPOSE:
    Defines the SQLite relational database schema using Flask-SQLAlchemy ORM:
      1. `User` (# [DATABASE], # [LOGIN]): Stores registered farmers/agronomists
         and administrators with hashed passwords and roles ("user" or "admin").
      2. `ScanRecord` (# [DATABASE]): Stores every leaf scan performed by a user,
         including the CNN Top-3 JSON, YOLOv8 bounding boxes JSON, affected leaf
         area %, Decision Tree severity, 8-element Feature Array, Hash Bucket ID,
         and Priority Queue ranked treatments.
      3. `FeedbackTicket` (# [DATABASE], # [FEEDBACK], # [ADMIN]): Stores user
         complaints/suggestions with complainant name, email, timestamp, tracking
         status, and the official response message sent by the Admin.

CONCEPTS USED:
    - Relational ORM Mapping with SQLite (# [DATABASE])
    - UserMixin Integration with Flask-Login (# [LOGIN])

HOW IT CONNECTS TO OTHER FILES:
    - Initialized with the Flask app inside `app.py`.
    - Queried and updated by `auth.py` and `app.py`.
================================================================================
"""

# Import datetime for human-readable timestamps and JSON for serializing lists/dicts
from datetime import datetime
import json
from typing import Any, Dict, List

# Import Flask-Login UserMixin for session management
from flask_login import UserMixin

# Import Flask-SQLAlchemy for ORM models
from flask_sqlalchemy import SQLAlchemy

# Instantiate the shared SQLAlchemy database object
db = SQLAlchemy()


# ==============================================================================
# 1. USER & ADMIN ACCOUNT TABLE
# ==============================================================================

# # [DATABASE] # [LOGIN] # [ADMIN]
class User(UserMixin, db.Model):
    """
    Represents a registered account in the SQLite `users` table.
    `role` is either `"user"` (farmer / field agronomist) or `"admin"` (project team).
    """

    __tablename__ = "users"

    # Primary key user ID
    id = db.Column(db.Integer, primary_key=True)
    # Full name of the user or administrator
    name = db.Column(db.String(120), nullable=False)
    # Unique login email address
    email = db.Column(db.String(150), unique=True, nullable=False, index=True)
    # Salted and hashed password string (via werkzeug.security)
    password_hash = db.Column(db.String(256), nullable=False)
    # Role string: 'user' or 'admin'
    role = db.Column(db.String(20), nullable=False, default="user")
    # Account creation timestamp (YYYY-MM-DD HH:MM:SS)
    created_at = db.Column(
        db.String(32),
        nullable=False,
        default=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    @property
    def is_admin(self) -> bool:
        """Returns True if this account has administrator privileges."""
        return self.role == "admin"


# ==============================================================================
# 2. LEAF SCAN HISTORY TABLE
# ==============================================================================

# # [DATABASE] # [CNN] # [YOLO]
class ScanRecord(db.Model):
    """
    Stores a single leaf scan result in the SQLite `scan_records` table,
    linking the deep learning outputs and 4-stage DSA values to the user who
    uploaded the image.
    """

    __tablename__ = "scan_records"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    user_name = db.Column(db.String(120), nullable=False)
    user_email = db.Column(db.String(150), nullable=False)

    # Uploaded raw image filename and YOLOv8 annotated image filename inside static/uploads/
    image_filename = db.Column(db.String(255), nullable=False)
    annotated_filename = db.Column(db.String(255), nullable=False)

    # Fused diagnosis metadata
    crop = db.Column(db.String(64), nullable=False)
    detected_pest = db.Column(db.String(120), nullable=False)
    scientific_name = db.Column(db.String(150), nullable=False, default="")
    fused_confidence_percent = db.Column(db.Float, nullable=False, default=0.0)
    models_agreed = db.Column(db.Boolean, nullable=False, default=True)
    winning_source = db.Column(db.String(80), nullable=False, default="CNN + YOLOv8")
    fusion_explanation = db.Column(db.Text, nullable=False, default="")
    is_uncertain = db.Column(db.Boolean, nullable=False, default=False)

    # Stage 1..4 DSA outputs stored for history & DSA Guide inspection
    affected_area_percent = db.Column(db.Float, nullable=False, default=0.0)
    severity = db.Column(db.String(32), nullable=False, default="Low")
    decision_rule = db.Column(db.Text, nullable=False, default="")
    hash_key_used = db.Column(db.String(150), nullable=False, default="")
    hash_bucket_index = db.Column(db.Integer, nullable=False, default=0)

    # JSON-serialized lists for CNN Top-3, YOLO boxes, 8-element Feature Array, and Ranked Treatments
    cnn_top3_json = db.Column(db.Text, nullable=False, default="[]")
    yolo_boxes_json = db.Column(db.Text, nullable=False, default="[]")
    feature_array_json = db.Column(db.Text, nullable=False, default="[]")
    ranked_treatments_json = db.Column(db.Text, nullable=False, default="[]")

    created_at = db.Column(
        db.String(32),
        nullable=False,
        default=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    @property
    def cnn_top3(self) -> List[Dict[str, Any]]:
        """Parses and returns the CNN Top-3 predictions list."""
        try:
            return json.loads(self.cnn_top3_json)
        except Exception:
            return []

    @property
    def yolo_boxes(self) -> List[Dict[str, Any]]:
        """Parses and returns the YOLOv8 bounding boxes list."""
        try:
            return json.loads(self.yolo_boxes_json)
        except Exception:
            return []

    @property
    def feature_array(self) -> List[float]:
        """Parses and returns the 8-element OpenCV feature array."""
        try:
            return json.loads(self.feature_array_json)
        except Exception:
            return [0.0] * 8

    @property
    def ranked_treatments(self) -> List[Dict[str, Any]]:
        """Parses and returns the Max-Heap Priority Queue ranked treatments list."""
        try:
            return json.loads(self.ranked_treatments_json)
        except Exception:
            return []


# ==============================================================================
# 3. USER FEEDBACK & COMPLAINT TRACKING TABLE
# ==============================================================================

# # [DATABASE] # [FEEDBACK] # [ADMIN]
class FeedbackTicket(db.Model):
    """
    Stores a user complaint or feedback ticket in the SQLite `feedback_tickets`
    table, including complainant name, email, submission date/time, tracking
    status, and the Admin's response message (`admin_reply`, `replied_by`, `replied_at`).
    """

    __tablename__ = "feedback_tickets"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    user_name = db.Column(db.String(120), nullable=False)
    user_email = db.Column(db.String(150), nullable=False)

    crop = db.Column(db.String(64), nullable=False, default="Rice")
    category = db.Column(db.String(64), nullable=False, default="Detection Issue")
    rating = db.Column(db.Integer, nullable=False, default=5)
    subject = db.Column(db.String(200), nullable=False)
    message = db.Column(db.Text, nullable=False)

    # Status is either 'Pending Admin Action' or 'Resolved ✓'
    status = db.Column(
        db.String(40), nullable=False, default="Pending Admin Action"
    )
    created_at = db.Column(
        db.String(32),
        nullable=False,
        default=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    # Admin response fields (visible in the user's account when they log in)
    admin_reply = db.Column(db.Text, nullable=True)
    replied_by = db.Column(db.String(120), nullable=True)
    replied_at = db.Column(db.String(32), nullable=True)
