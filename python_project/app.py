"""
================================================================================
FILE: app.py
PURPOSE:
    Main Flask web application server for:
    "SMART PEST DETECTION AND TREATMENT RECOMMENDATION SYSTEM USING
     COMPUTER VISION AND MACHINE LEARNING"
    Project By: PHANI NANDAN M S, VEDHESHA PARTHIBAN, KHADGA LOHITH K, SHIVANI BALLANI

    ROUTES IMPLEMENTED:
      - `GET /` & `GET, POST /scanner`: Upload a leaf image, run real CNN + YOLOv8
        inference and the 4-stage DSA pipeline, and display bounding boxes,
        CNN Top-3 probabilities, severity, and ranked treatments.
      - `GET /history` & `GET /history/export_csv`: View and export per-user
        scan history as CSV.
      - `GET, POST /feedback`: Submit detection/treatment complaints, track
        status with a 3-step progress bar, and read official Admin responses.
      - `GET /dsa-guide`: View the 4 DSA stages populated with the real values
        from the user's most recent scan.
      - `GET /admin` & `POST /admin/reply/<id>`: Admin Portal showing all users,
        all scans, pest-frequency distribution, model confidence statistics,
        the 31-bucket Hash Table inspector, and direct complaint response messaging.
      - `GET /api/health`: JSON health check endpoint.

CONCEPTS USED:
    - Flask + Jinja2 Server-Side Rendering (# [API_ROUTE])
    - SQLite Persistence via SQLAlchemy (# [DATABASE])
    - Real CNN + YOLOv8 + 4-Stage DSA Integration (# [CNN], # [YOLO], # [ADMIN])

HOW TO RUN:
    pip install -r requirements.txt
    python app.py
================================================================================
"""

# Import standard library modules
from collections import Counter
from datetime import datetime
import io
import csv
import json
import os
import uuid
from typing import Any, Dict

# Import Flask modules
from flask import (
    Flask,
    Response,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_login import LoginManager, current_user, login_required
from werkzeug.utils import secure_filename

# Import project modules
import config
from auth import auth_bp, seed_default_accounts
from data.pest_database import GLOBAL_PEST_HASH_TABLE, IP102_TOP20_PEST_DATABASE
from database import FeedbackTicket, ScanRecord, User, db
from dsa_pipeline import SeverityDecisionTree
from inference import analyze_leaf_image
from models_loader import MODEL_REGISTRY


# ==============================================================================
# 1. FLASK APPLICATION FACTORY & EXTENSION INITIALIZATION
# ==============================================================================

def create_app() -> Flask:
    """Creates and configures the Flask web application and SQLite database."""
    app = Flask(__name__)
    app.config["SECRET_KEY"] = config.SECRET_KEY
    app.config["SQLALCHEMY_DATABASE_URI"] = config.SQLALCHEMY_DATABASE_URI
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = config.SQLALCHEMY_TRACK_MODIFICATIONS
    app.config["UPLOAD_FOLDER"] = config.UPLOAD_FOLDER

    # Ensure upload and model directories exist
    os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    # Initialize SQLAlchemy and Flask-Login
    db.init_app(app)
    login_manager = LoginManager()
    login_manager.login_view = "auth.login"
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id: str):
        return db.session.get(User, int(user_id))

    # Inject global project title, team names, and unread reply count into all Jinja2 templates
    @app.context_processor
    def inject_global_template_vars() -> Dict[str, Any]:
        user_tickets_count = 0
        user_replies_count = 0
        pending_admin_count = 0
        if current_user.is_authenticated:
            my_tickets = FeedbackTicket.query.filter_by(user_id=current_user.id).all()
            user_tickets_count = len(my_tickets)
            user_replies_count = sum(1 for t in my_tickets if t.admin_reply)
            pending_admin_count = FeedbackTicket.query.filter_by(
                status="Pending Admin Action"
            ).count()
        return {
            "PROJECT_TITLE": config.PROJECT_TITLE,
            "PROJECT_BY": config.PROJECT_BY,
            "user_tickets_count": user_tickets_count,
            "user_replies_count": user_replies_count,
            "pending_admin_count": pending_admin_count,
            "model_status": {
                "cnn_ready": MODEL_REGISTRY.cnn_model is not None,
                "yolo_ready": MODEL_REGISTRY.yolo_model is not None,
                "cnn_status": MODEL_REGISTRY.cnn_status,
                "yolo_status": MODEL_REGISTRY.yolo_status,
            },
        }

    # Register authentication blueprint
    app.register_blueprint(auth_bp)

    # Create tables and seed default Admin & Farmer accounts
    with app.app_context():
        db.create_all()
        seed_default_accounts()

    # --------------------------------------------------------------------------
    # ROUTE 1: ROOT & LEAF SCANNER PAGE (/ and /scanner)
    # --------------------------------------------------------------------------
    @app.route("/")
    def index():
        if not current_user.is_authenticated:
            return redirect(url_for("auth.login"))
        return redirect(url_for("scanner_page"))

    # # [API_ROUTE] # [CNN] # [YOLO]
    @app.route("/scanner", methods=["GET", "POST"])
    @login_required
    def scanner_page():
        """
        Handles leaf image uploads, executes real CNN + YOLOv8 + 4-stage DSA
        inference, stores the result in SQLite, and displays the full diagnosis.
        """
        scan_result = None
        uploaded_image_url = None
        annotated_image_url = None

        if request.method == "POST":
            file = request.files.get("leaf_image")
            if file is None or file.filename == "":
                flash("Please choose a leaf image file (JPG/PNG/WEBP) to scan.", "error")
                return redirect(url_for("scanner_page"))

            ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
            if ext not in config.ALLOWED_EXTENSIONS:
                flash("Unsupported file format. Please upload JPG, JPEG, PNG, or WEBP.", "error")
                return redirect(url_for("scanner_page"))

            # Generate unique filenames for raw upload and YOLOv8 annotated output
            unique_token = uuid.uuid4().hex[:10]
            safe_base = secure_filename(file.filename.rsplit(".", 1)[0]) or "leaf"
            raw_filename = f"{safe_base}_{unique_token}.{ext}"
            annotated_filename = f"{safe_base}_{unique_token}_yolo.{ext}"

            raw_path = os.path.join(config.UPLOAD_FOLDER, raw_filename)
            annotated_path = os.path.join(config.UPLOAD_FOLDER, annotated_filename)
            file.save(raw_path)

            # Run real inference pipeline
            scan_result = analyze_leaf_image(raw_path, annotated_output_path=annotated_path)
            uploaded_image_url = url_for("static", filename=f"uploads/{raw_filename}")
            if os.path.isfile(annotated_path):
                annotated_image_url = url_for(
                    "static", filename=f"uploads/{annotated_filename}"
                )
            else:
                annotated_image_url = uploaded_image_url

            # Save scan to SQLite database if models were trained and ran
            if scan_result.get("status") in ("ok", "uncertain"):
                record = ScanRecord(
                    user_id=current_user.id,
                    user_name=current_user.name,
                    user_email=current_user.email,
                    image_filename=raw_filename,
                    annotated_filename=(
                        annotated_filename
                        if os.path.isfile(annotated_path)
                        else raw_filename
                    ),
                    crop=scan_result.get("crop", "Unknown"),
                    detected_pest=scan_result.get("display_name", "Uncertain"),
                    scientific_name=scan_result.get("scientific_name", ""),
                    fused_confidence_percent=scan_result.get(
                        "fused_confidence_percent", 0.0
                    ),
                    models_agreed=bool(scan_result.get("models_agreed", True)),
                    winning_source=scan_result.get("winning_source", ""),
                    fusion_explanation=scan_result.get("fusion_explanation", ""),
                    is_uncertain=bool(scan_result.get("is_uncertain", False)),
                    affected_area_percent=scan_result.get("affected_area_percent", 0.0),
                    severity=scan_result.get("severity", "Low"),
                    decision_rule=scan_result.get("decision_rule", ""),
                    hash_key_used=scan_result.get("hash_key_used", ""),
                    hash_bucket_index=int(scan_result.get("hash_bucket_index", 0)),
                    cnn_top3_json=json.dumps(scan_result.get("cnn_top3", [])),
                    yolo_boxes_json=json.dumps(scan_result.get("yolo_boxes", [])),
                    feature_array_json=json.dumps(scan_result.get("feature_array", [])),
                    ranked_treatments_json=json.dumps(
                        scan_result.get("ranked_treatments", [])
                    ),
                )
                db.session.add(record)
                db.session.commit()

        # Fetch the user's most recent scans for quick reference
        recent_scans = (
            ScanRecord.query.filter_by(user_id=current_user.id)
            .order_by(ScanRecord.id.desc())
            .limit(5)
            .all()
        )

        return render_template(
            "scanner.html",
            scan_result=scan_result,
            uploaded_image_url=uploaded_image_url,
            annotated_image_url=annotated_image_url,
            recent_scans=recent_scans,
            confidence_cutoff_pct=int(config.CONFIDENCE_CUTOFF * 100),
            supported_pests=IP102_TOP20_PEST_DATABASE,
        )

    # --------------------------------------------------------------------------
    # ROUTE 2: USER SCAN HISTORY & CSV EXPORT (/history)
    # --------------------------------------------------------------------------
    # # [API_ROUTE] # [DATABASE]
    @app.route("/history")
    @login_required
    def history_page():
        """Displays all past leaf scans belonging to the currently logged-in user."""
        q = (request.args.get("q") or "").strip().lower()
        scans_query = ScanRecord.query.filter_by(user_id=current_user.id).order_by(
            ScanRecord.id.desc()
        )
        all_user_scans = scans_query.all()

        if q:
            filtered_scans = [
                s
                for s in all_user_scans
                if q in s.crop.lower()
                or q in s.detected_pest.lower()
                or q in s.severity.lower()
            ]
        else:
            filtered_scans = all_user_scans

        return render_template("history.html", scans=filtered_scans, search_query=q)

    # # [API_ROUTE]
    @app.route("/history/export_csv")
    @login_required
    def export_history_csv():
        """Exports the logged-in user's scan history as a downloadable CSV file."""
        user_scans = (
            ScanRecord.query.filter_by(user_id=current_user.id)
            .order_by(ScanRecord.id.desc())
            .all()
        )
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            [
                "Scan ID",
                "Timestamp",
                "Crop",
                "Detected Pest",
                "Confidence %",
                "Affected Area %",
                "Severity",
                "Hash Bucket",
                "Rank #1 Treatment",
            ]
        )
        for s in user_scans:
            top_tx = (
                s.ranked_treatments[0]["name"] if s.ranked_treatments else "N/A"
            )
            writer.writerow(
                [
                    f"SCAN-{s.id}",
                    s.created_at,
                    s.crop,
                    s.detected_pest,
                    s.fused_confidence_percent,
                    s.affected_area_percent,
                    s.severity,
                    s.hash_bucket_index,
                    top_tx,
                ]
            )

        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=my_pest_scan_history.csv"
            },
        )

    # --------------------------------------------------------------------------
    # ROUTE 3: USER FEEDBACK, COMPLAINT TRACKER & ADMIN RESPONSE INBOX (/feedback)
    # --------------------------------------------------------------------------
    # # [API_ROUTE] # [FEEDBACK]
    @app.route("/feedback", methods=["GET", "POST"])
    @login_required
    def feedback_page():
        """
        Allows users to submit a complaint/feedback ticket, track their submitted
        tickets with a 3-step live progress bar, and view responses sent by the Admin.
        """
        if request.method == "POST":
            crop = (request.form.get("crop") or "Rice").strip()
            category = (request.form.get("category") or "Detection Issue").strip()
            rating = int(request.form.get("rating") or 5)
            subject = (request.form.get("subject") or "").strip()
            message = (request.form.get("message") or "").strip()

            if not subject or not message:
                flash("Please provide both a subject title and detailed description.", "error")
                return redirect(url_for("feedback_page"))

            ticket = FeedbackTicket(
                user_id=current_user.id,
                user_name=current_user.name,
                user_email=current_user.email,
                crop=crop,
                category=category,
                rating=max(1, min(5, rating)),
                subject=subject,
                message=message,
                status="Pending Admin Action",
                created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
            db.session.add(ticket)
            db.session.commit()
            flash(
                "Complaint ticket submitted! You can track its live status and Admin response below.",
                "success",
            )
            return redirect(url_for("feedback_page"))

        view_filter = request.args.get("filter", "my")
        my_tickets = (
            FeedbackTicket.query.filter_by(user_id=current_user.id)
            .order_by(FeedbackTicket.id.desc())
            .all()
        )
        all_tickets = FeedbackTicket.query.order_by(FeedbackTicket.id.desc()).all()

        displayed_tickets = my_tickets if view_filter == "my" else all_tickets
        crops_list = sorted({item["crop"] for item in IP102_TOP20_PEST_DATABASE})

        return render_template(
            "feedback.html",
            tickets=displayed_tickets,
            my_tickets_count=len(my_tickets),
            all_tickets_count=len(all_tickets),
            view_filter=view_filter,
            crops_list=crops_list,
        )

    # --------------------------------------------------------------------------
    # ROUTE 4: 4-STAGE DSA GUIDE PAGE WITH REAL VALUES FROM LAST SCAN (/dsa-guide)
    # --------------------------------------------------------------------------
    # # [API_ROUTE] # [ARRAY] # [HASH_TABLE] # [DECISION_TREE] # [PRIORITY_QUEUE]
    @app.route("/dsa-guide")
    @login_required
    def dsa_guide_page():
        """
        Displays the 4 Data Structure stages (Feature Array, 31-Bucket Hash Table,
        Decision Tree, Max-Heap Priority Queue) populated with the real values
        from the user's most recent scan (or the latest system scan).
        """
        last_scan = (
            ScanRecord.query.filter_by(user_id=current_user.id)
            .order_by(ScanRecord.id.desc())
            .first()
        )
        if last_scan is None:
            last_scan = ScanRecord.query.order_by(ScanRecord.id.desc()).first()

        # Interactive Decision Tree threshold test from query string (?test_area=24)
        test_area = float(request.args.get("test_area", 24.0))
        tree_sim = SeverityDecisionTree().evaluate(test_area)

        return render_template(
            "dsa_guide.html",
            last_scan=last_scan,
            feature_names=config.FEATURE_NAMES,
            hash_buckets=GLOBAL_PEST_HASH_TABLE.get_all_bucket_stats(),
            test_area=test_area,
            tree_sim=tree_sim,
            config=config,
        )

    # --------------------------------------------------------------------------
    # ROUTE 5: ADMIN PORTAL & DIRECT COMPLAINT RESPONSE MESSAGING (/admin)
    # --------------------------------------------------------------------------
    # # [API_ROUTE] # [ADMIN]
    @app.route("/admin")
    @login_required
    def admin_page():
        """
        Admin dashboard displaying all users, all scans, pest-frequency chart data,
        model confidence statistics, the feedback inbox with complainant identity &
        response composer, and the 31-bucket Hash Table database inspector.
        """
        if not current_user.is_admin:
            flash("Access restricted to verified Project Administrators.", "error")
            return redirect(url_for("scanner_page"))

        all_users = User.query.order_by(User.id.desc()).all()
        all_scans = ScanRecord.query.order_by(ScanRecord.id.desc()).all()
        all_tickets = FeedbackTicket.query.order_by(FeedbackTicket.id.desc()).all()

        # Compute pest-frequency distribution across all scans
        pest_counts = Counter(s.detected_pest for s in all_scans)
        total_scans_count = max(1, len(all_scans))
        pest_frequency_stats = [
            {
                "pest_name": pest,
                "count": count,
                "percent": round((count / total_scans_count) * 100.0, 1),
            }
            for pest, count in pest_counts.most_common()
        ]

        # Compute model confidence statistics (mean, min, max, uncertain count)
        confidences = [s.fused_confidence_percent for s in all_scans if s.fused_confidence_percent > 0]
        confidence_stats = {
            "mean": round(sum(confidences) / len(confidences), 2) if confidences else 0.0,
            "max": round(max(confidences), 2) if confidences else 0.0,
            "min": round(min(confidences), 2) if confidences else 0.0,
            "uncertain_count": sum(1 for s in all_scans if s.is_uncertain),
            "agreement_rate": (
                round(
                    (sum(1 for s in all_scans if s.models_agreed) / len(all_scans)) * 100.0,
                    1,
                )
                if all_scans
                else 100.0
            ),
        }

        return render_template(
            "admin.html",
            all_users=all_users,
            all_scans=all_scans,
            all_tickets=all_tickets,
            pest_frequency_stats=pest_frequency_stats,
            confidence_stats=confidence_stats,
            pest_database=IP102_TOP20_PEST_DATABASE,
            hash_buckets=GLOBAL_PEST_HASH_TABLE.get_all_bucket_stats(),
        )

    # # [API_ROUTE] # [ADMIN] # [FEEDBACK]
    @app.route("/admin/reply/<int:ticket_id>", methods=["POST"])
    @login_required
    def admin_reply_ticket(ticket_id: int):
        """
        Saves the Admin's custom response message to a user's complaint ticket,
        records the Admin's name and timestamp, and marks the ticket 'Resolved ✓'.
        """
        if not current_user.is_admin:
            flash("Unauthorized action.", "error")
            return redirect(url_for("scanner_page"))

        ticket = db.session.get(FeedbackTicket, ticket_id)
        if ticket is None:
            flash("Complaint ticket not found.", "error")
            return redirect(url_for("admin_page"))

        reply_text = (request.form.get("admin_reply") or "").strip()
        if not reply_text:
            reply_text = (
                f"Hello {ticket.user_name}, your complaint regarding {ticket.crop} "
                f"({ticket.subject}) has been reviewed and resolved by the Project Team."
            )

        ticket.admin_reply = reply_text
        ticket.replied_by = f"{current_user.name} (Admin)"
        ticket.replied_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ticket.status = "Resolved ✓"
        db.session.commit()

        flash(
            f"Response sent to {ticket.user_name} ({ticket.user_email}) for Ticket #{ticket.id}!",
            "success",
        )
        return redirect(url_for("admin_page"))

    # --------------------------------------------------------------------------
    # ROUTE 6: JSON HEALTH CHECK ENDPOINT (/api/health)
    # --------------------------------------------------------------------------
    # # [API_ROUTE]
    @app.route("/api/health")
    def api_health():
        """Returns JSON health status of Flask, SQLite DB, CNN, YOLOv8, and Hash Table."""
        return jsonify(
            {
                "status": "healthy",
                "project": config.PROJECT_TITLE,
                "project_by": config.PROJECT_BY,
                "cnn_ready": MODEL_REGISTRY.cnn_model is not None,
                "cnn_status": MODEL_REGISTRY.cnn_status,
                "yolo_ready": MODEL_REGISTRY.yolo_model is not None,
                "yolo_status": MODEL_REGISTRY.yolo_status,
                "hash_table_buckets": config.HASH_TABLE_BUCKETS,
                "top_n_classes": config.TOP_N_CLASSES,
            }
        )

    return app


# Create global Flask app instance
app = create_app()

if __name__ == "__main__":
    print("=" * 78)
    print(config.PROJECT_TITLE)
    print(f"Project By: {config.PROJECT_BY}")
    print(f"Starting Flask Server on http://127.0.0.1:{config.PORT}")
    print("=" * 78)
    app.run(host="0.0.0.0", port=config.PORT, debug=True)
