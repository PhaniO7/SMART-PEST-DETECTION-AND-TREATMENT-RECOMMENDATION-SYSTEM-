"""
================================================================================
FILE: auth.py
PURPOSE:
    Handles User and Admin authentication using Flask-Login and Werkzeug
    password hashing (`generate_password_hash`, `check_password_hash`):
      - `/login` (GET, POST): Authenticates existing users/admins or handles
        one-click quick demo login for project evaluation.
      - `/signup` (POST): Registers a new user account and automatically assigns
        the `"admin"` role if the email belongs to `config.ADMIN_EMAILS`.
      - `/logout` (GET): Ends the current user session.
      - `seed_default_accounts()`: Creates default Admin and Farmer accounts plus
        sample feedback tickets on first run so the database is ready immediately.

CONCEPTS USED:
    - Salted Password Hashing (# [LOGIN])
    - Role-Based Access Control ("user" vs "admin") (# [ADMIN])

HOW IT CONNECTS TO OTHER FILES:
    - Uses `db`, `User`, and `FeedbackTicket` from `database.py`.
    - Registered as a Flask Blueprint inside `app.py`.
================================================================================
"""

# Import Flask utilities for routing, forms, flash messages, and redirects
from flask import Blueprint, flash, redirect, render_template, request, url_for

# Import Flask-Login functions for session login/logout
from flask_login import current_user, login_required, login_user, logout_user

# Import Werkzeug password hashing functions
from werkzeug.security import check_password_hash, generate_password_hash

# Import central config and database models
import config
from database import FeedbackTicket, User, db

# Create the authentication Blueprint
auth_bp = Blueprint("auth", __name__)


# ==============================================================================
# 1. SEED DEFAULT ADMIN & FARMER ACCOUNTS ON FIRST DATABASE INITIALIZATION
# ==============================================================================

# # [LOGIN] # [ADMIN] # [DATABASE]
def seed_default_accounts() -> None:
    """
    Seeds default Admin (`phaninandan78@gmail.com`) and Farmer
    (`ramesh.kumar@icar.gov.in`) accounts along with two sample feedback tickets
    if the `users` table is empty.
    """
    if User.query.first() is not None:
        return

    # 1. Default Project Team Admin Account
    admin_user = User(
        name="PHANI NANDAN M S",
        email="phaninandan78@gmail.com",
        password_hash=generate_password_hash("admin123"),
        role="admin",
    )
    # 2. Default Field Agronomist / Farmer Account
    farmer_user = User(
        name="Dr. Ramesh Kumar",
        email="ramesh.kumar@icar.gov.in",
        password_hash=generate_password_hash("farmer123"),
        role="user",
    )
    db.session.add_all([admin_user, farmer_user])
    db.session.commit()

    # Seed two initial feedback/complaint tickets so the tracker & admin inbox are populated
    tkt1 = FeedbackTicket(
        user_id=farmer_user.id,
        user_name=farmer_user.name,
        user_email=farmer_user.email,
        crop="Rice",
        category="Treatment Feedback",
        rating=5,
        subject="Water level during Pymetrozine 50% WG spray for Rice Brown Planthopper",
        message=(
            "Recommended dosage of 120 g/acre worked well. Should we maintain 2 cm "
            "standing water in the paddy field during application?"
        ),
        status="Resolved ✓",
        created_at="2026-10-21 11:20:15",
        admin_reply=(
            "Yes, maintain 2–3 cm standing water for 48 hours and direct the spray "
            "nozzle toward the base of the rice tillers."
        ),
        replied_by="PHANI NANDAN M S (Admin)",
        replied_at="2026-10-21 14:05:30",
    )
    tkt2 = FeedbackTicket(
        user_id=farmer_user.id,
        user_name=farmer_user.name,
        user_email=farmer_user.email,
        crop="Corn",
        category="Detection Issue",
        rating=4,
        subject="Shaded lower maize leaf with armyworm frass",
        message=(
            "When photographing a heavily shaded lower maize whorl leaf in late evening, "
            "confidence was close to the threshold. Any lighting tips?"
        ),
        status="Pending Admin Action",
        created_at="2026-10-22 16:05:42",
    )
    db.session.add_all([tkt1, tkt2])
    db.session.commit()


# ==============================================================================
# 2. LOGIN, SIGN-UP & LOGOUT ROUTES
# ==============================================================================

# # [LOGIN] # [API_ROUTE]
@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    """Renders the Login/Sign-Up screen and authenticates credentials via POST."""
    if current_user.is_authenticated:
        return redirect(url_for("scanner_page"))

    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""

        user = User.query.filter_by(email=email).first()
        if user is None or not check_password_hash(user.password_hash, password):
            flash("Invalid email address or password. Please try again.", "error")
            return redirect(url_for("auth.login"))

        login_user(user)
        flash(f"Welcome back, {user.name} ({user.role.upper()})!", "success")
        return redirect(url_for("scanner_page"))

    return render_template("login.html")


# # [LOGIN] # [ADMIN] # [API_ROUTE]
@auth_bp.route("/signup", methods=["POST"])
def signup():
    """Creates a new user account with a hashed password and logs them in."""
    name = (request.form.get("name") or "").strip()
    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""

    if not name or not email or not password:
        flash("Please fill in your full name, email address, and password.", "error")
        return redirect(url_for("auth.login"))

    existing = User.query.filter_by(email=email).first()
    if existing is not None:
        flash("An account with that email already exists. Please log in.", "error")
        return redirect(url_for("auth.login"))

    # Assign 'admin' role if email matches config.ADMIN_EMAILS or team member names
    is_team_admin = (
        email in config.ADMIN_EMAILS
        or any(
            member in name.lower()
            for member in ("phani", "vedhesha", "khadga", "shivani")
        )
    )
    role = "admin" if is_team_admin else "user"

    new_user = User(
        name=name,
        email=email,
        password_hash=generate_password_hash(password),
        role=role,
    )
    db.session.add(new_user)
    db.session.commit()

    login_user(new_user)
    flash(f"Account created successfully! Logged in as {new_user.name} ({role.upper()}).", "success")
    return redirect(url_for("scanner_page"))


# # [LOGIN] # [API_ROUTE]
@auth_bp.route("/quick-login/<role_type>", methods=["POST"])
def quick_login(role_type: str):
    """One-click quick login helper for Admin or Farmer demo evaluation."""
    target_email = (
        "phaninandan78@gmail.com"
        if role_type == "admin"
        else "ramesh.kumar@icar.gov.in"
    )
    user = User.query.filter_by(email=target_email).first()
    if user is not None:
        login_user(user)
        flash(f"Signed in as {user.name} ({user.role.upper()}).", "success")
        return redirect(url_for("scanner_page"))

    flash("Quick login account not found.", "error")
    return redirect(url_for("auth.login"))


# # [LOGIN] # [API_ROUTE]
@auth_bp.route("/logout")
@login_required
def logout():
    """Logs out the current user and redirects to the login page."""
    logout_user()
    flash("You have been signed out.", "info")
    return redirect(url_for("auth.login"))
