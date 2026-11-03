"""
================================================================================
FILE: app.py
PURPOSE:
    The complete 100% Python Streamlit web application for:
    "Smart Pest Detection and Treatment Recommendation System using Computer
    Vision and Machine Learning" (B.Tech AI & DS Design Project).

    Includes 6 integrated views:
      1. Login & Registration (`# [LOGIN]`) — Empty inputs (no pre-filled email
         or password). Demo quick-login buttons only appear if
         `config.DEMO_MODE == True` (default is `False`).
      2. Leaf Pest Scanner (`# [CNN]` `# [YOLO]`) — Runs real PyTorch
         `EfficientNet-B0` + Ultralytics `YOLOv8n` inference and the 4-stage DSA
         pipeline timed with `time.perf_counter()`.
      3. Scan History (`# [DATABASE]`) — Displays saved scans and measured DSA
         pipeline execution speed (`ms`).
      4. Feedback & Complaint Tracker (`# [FEEDBACK]`) — Users submit complaints
         and view admin response messages and status inside their account.
      5. Admin Dashboard (`# [ADMIN]`) — Admins see who made each complaint
         (name, email, date and time) and send response messages back to users.
      6. DSA Guide & Live Inspector (`# [ARRAY]` `# [HASH_TABLE]`
         `# [DECISION_TREE]` `# [PRIORITY_QUEUE]`) — Interactive walkthrough of
         all 4 data structures with Big-O complexity analysis.

    ZERO HARDCODED STATS:
      - Crop & pest counts always use `config.NUM_CROPS` and `config.NUM_PESTS`
        (`len(config.CROPS)` and `len(config.PEST_SHORTLIST)`).
      - "Detection Accuracy" reads the real test-set accuracy saved to SQLite by
        `evaluate.py`, or shows `"Not evaluated yet"` if not evaluated yet.
      - "DSA Pipeline Speed" shows the real measured `time.perf_counter()` speed
        of the last scan (and average over saved scans), or `"No scans yet"`.

HOW IT CONNECTS TO OTHER FILES:
    - Imports `config.py`, `pest_data.py`, `dsa.py`, `detect.py`, and `database.py`.
    - Run locally in VS Code with: `streamlit run app.py`
================================================================================
"""

# Import standard library modules
import os
import time
from typing import Any, Dict, List, Optional

# Import project modules
import config
import database
import detect
import dsa
from pest_data import PEST_DATABASE, get_pest_by_class_id


# ==============================================================================
# 1. DYNAMIC METRIC HELPERS (ZERO HARDCODED STATS) (# [CONFIG] # [DATABASE])
# ==============================================================================

# # [CONFIG] # [DATABASE]
def get_header_scope_text() -> str:
    """
    Returns the dynamic scope text `"{NUM_CROPS} crops and {NUM_PESTS} pests"`
    computed with `len()` from `config.py` so it can never be out of date.
    """
    return f"{config.NUM_CROPS} crops and {config.NUM_PESTS} pests"


# # [DATABASE] # [CNN] # [YOLO]
def get_live_dashboard_stats(
    last_scan_speed_ms: Optional[float] = None,
    db_path: Optional[str] = None,
) -> Dict[str, str]:
    """
    Computes all 4 hero stat card values dynamically from `config.py` and the
    SQLite database (`database.py`):
      - `crops_text`: `f"{config.NUM_CROPS} crops"`
      - `pests_text`: `f"{len(config.PEST_SHORTLIST)} pests"`
      - `accuracy_text`: Real CNN test-set accuracy from `evaluate.py` in SQLite,
        or `"Not evaluated yet"` if `evaluate.py` has not been run yet.
      - `dsa_speed_text`: Measured time (ms) of the last scan or average over
        saved scans in SQLite, or `"No scans yet"` when no scans exist.
    """
    eval_row = database.get_latest_evaluation_metrics(db_path=db_path)
    if eval_row is not None and eval_row.get("cnn_test_accuracy") is not None:
        acc_val = float(eval_row["cnn_test_accuracy"])
        f1_val = float(eval_row.get("cnn_macro_f1", 0.0))
        yolo_val = eval_row.get("yolo_map50")
        eval_date = str(eval_row.get("evaluated_at", ""))
        accuracy_text = f"{acc_val:.2f}%"
        if yolo_val is not None:
            accuracy_caption = (
                f"Macro F1: {f1_val:.2f}% | YOLO mAP50: {float(yolo_val):.2f}% ({eval_date})"
            )
        else:
            accuracy_caption = f"Macro F1: {f1_val:.2f}% (Evaluated {eval_date})"
    else:
        accuracy_text = "Not evaluated yet"
        accuracy_caption = "Run evaluate.py after training to measure real accuracy"

    speed_stats = database.get_dsa_speed_stats(db_path=db_path)
    if last_scan_speed_ms is not None:
        dsa_speed_text = f"{float(last_scan_speed_ms):.2f} ms"
        if speed_stats["has_scans"]:
            dsa_speed_caption = (
                f"Last scan measured via time.perf_counter() "
                f"(avg {speed_stats['avg_speed_ms']:.2f} ms over {speed_stats['total_scans']} scans)"
            )
        else:
            dsa_speed_caption = "Last scan measured via time.perf_counter()"
    elif speed_stats["has_scans"]:
        dsa_speed_text = f"{speed_stats['last_speed_ms']:.2f} ms"
        dsa_speed_caption = (
            f"Average: {speed_stats['avg_speed_ms']:.2f} ms across "
            f"{speed_stats['total_scans']} saved scans"
        )
    else:
        dsa_speed_text = "No scans yet"
        dsa_speed_caption = "Measured with time.perf_counter() during leaf scanning"

    return {
        "crops_text": f"{config.NUM_CROPS} crops",
        "pests_text": f"{len(config.PEST_SHORTLIST)} pests",
        "accuracy_text": accuracy_text,
        "accuracy_caption": accuracy_caption,
        "dsa_speed_text": dsa_speed_text,
        "dsa_speed_caption": dsa_speed_caption,
    }


# ==============================================================================
# 2. STREAMLIT UI SECTIONS (# [LOGIN] # [CNN] # [YOLO] # [FEEDBACK] # [ADMIN])
# ==============================================================================

# # [LOGIN]
def render_login_page(st: Any) -> None:
    """
    Renders the Login and Create Account page.
    - Email and password inputs are empty by default (never pre-filled).
    - Quick-access demo accounts only appear if `config.DEMO_MODE` is True.
    """
    st.title(config.PROJECT_TITLE)
    st.caption(
        f"Team: {config.PROJECT_TEAM} | Scope: {get_header_scope_text()} (IP102 Dataset)"
    )
    st.markdown("---")

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        tab_login, tab_register = st.tabs(["Sign In", "Create New Account"])

        with tab_login:
            st.subheader("Sign In to Your Account")
            with st.form("login_form", clear_on_submit=False):
                # Empty default values — no pre-filled email or password
                email_in = st.text_input(
                    "Email Address",
                    value="",
                    placeholder="Enter your email address",
                )
                password_in = st.text_input(
                    "Password",
                    value="",
                    type="password",
                    placeholder="Enter your password",
                )
                submitted = st.form_submit_button("Sign In", use_container_width=True)

            if submitted:
                ok, msg, user_obj = database.authenticate_user(email_in, password_in)
                if ok and user_obj is not None:
                    st.session_state["user"] = user_obj
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

        with tab_register:
            st.subheader("Register a Farmer / Student Account")
            with st.form("register_form", clear_on_submit=True):
                reg_name = st.text_input(
                    "Full Name", value="", placeholder="Enter your full name"
                )
                reg_email = st.text_input(
                    "Email Address", value="", placeholder="Enter your email address"
                )
                reg_pw = st.text_input(
                    "Create Password",
                    value="",
                    type="password",
                    placeholder="At least 4 characters",
                )
                reg_submit = st.form_submit_button(
                    "Create Account", use_container_width=True
                )

            if reg_submit:
                ok, msg, user_obj = database.register_user(
                    reg_name, reg_email, reg_pw
                )
                if ok and user_obj is not None:
                    st.session_state["user"] = user_obj
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

        # Quick-access demo accounts ONLY appear if config.DEMO_MODE is True
        if config.DEMO_MODE:
            st.markdown("---")
            st.info(
                "DEMO_MODE is enabled in config.py. Set DEMO_MODE = False before "
                "showing or sharing the project."
            )
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                if st.button("Quick Demo Farmer Login", use_container_width=True):
                    ok, _, u = database.authenticate_user(
                        config.DEMO_USER_EMAIL, config.DEMO_DEFAULT_PASSWORD
                    )
                    if ok and u:
                        st.session_state["user"] = u
                        st.rerun()
            with dcol2:
                if st.button("Quick Demo Admin Login", use_container_width=True):
                    ok, _, u = database.authenticate_user(
                        config.DEMO_ADMIN_EMAIL, config.DEMO_DEFAULT_PASSWORD
                    )
                    if ok and u:
                        st.session_state["user"] = u
                        st.rerun()

    with col_right:
        stats = get_live_dashboard_stats(
            last_scan_speed_ms=st.session_state.get("last_dsa_speed_ms")
        )
        st.subheader(f"System Overview ({get_header_scope_text()})")
        m1, m2 = st.columns(2)
        m1.metric("Supported Crops", stats["crops_text"])
        m2.metric("Pests Indexed", stats["pests_text"])

        m3, m4 = st.columns(2)
        m3.metric(
            "Detection Accuracy",
            stats["accuracy_text"],
            help=stats["accuracy_caption"],
        )
        m4.metric(
            "DSA Pipeline Speed",
            stats["dsa_speed_text"],
            help=stats["dsa_speed_caption"],
        )
        st.caption(stats["accuracy_caption"])
        st.caption(stats["dsa_speed_caption"])


# # [CNN] # [YOLO] # [ARRAY] # [HASH_TABLE] # [DECISION_TREE] # [PRIORITY_QUEUE]
def render_scanner_page(st: Any, user: Dict[str, Any]) -> None:
    """
    Renders the main Leaf Pest Scanner page, hero stats, CNN + YOLO results,
    4-stage DSA output, and the supported `{NUM_CROPS} crops and {NUM_PESTS} pests`
    reference table.
    """
    import cv2
    import numpy as np
    from PIL import Image

    st.header(
        f"Leaf Pest Scanner — Supporting {get_header_scope_text()}"
    )

    # Top dynamic stat cards (computed from config.py and SQLite)
    stats = get_live_dashboard_stats(
        last_scan_speed_ms=st.session_state.get("last_dsa_speed_ms")
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Supported Crops", stats["crops_text"])
    c2.metric("Pests Indexed", stats["pests_text"])
    c3.metric("Detection Accuracy", stats["accuracy_text"])
    c4.metric("DSA Pipeline Speed", stats["dsa_speed_text"])
    st.caption(f"{stats['accuracy_caption']}  |  {stats['dsa_speed_caption']}")

    st.markdown("---")

    col_in, col_out = st.columns([1, 1.3])

    with col_in:
        selected_crop = st.selectbox(
            f"Select Host Crop ({config.NUM_CROPS} crops supported)",
            options=config.CROPS,
        )
        uploaded_file = st.file_uploader(
            "Upload a field leaf/pest photo (JPG or PNG)",
            type=["jpg", "jpeg", "png"],
        )

        # Filter pests belonging to the selected crop for optional DSA-only testing
        crop_pests = [
            p for p in config.PEST_SHORTLIST if p["crop"] == selected_crop
        ]
        run_btn = st.button(
            "Run CNN + YOLOv8 + 4-Stage DSA Scan",
            type="primary",
            use_container_width=True,
            disabled=(uploaded_file is None),
        )

    with col_out:
        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file).convert("RGB")
            rgb_arr = np.array(pil_img)
            bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
            st.image(pil_img, caption="Uploaded Leaf Image", use_container_width=True)

            if run_btn:
                result = detect.analyze_leaf_image(
                    image_bgr=bgr_arr,
                    pil_image=pil_img,
                    selected_crop=selected_crop,
                )
                st.session_state["last_dsa_speed_ms"] = result["dsa_speed_ms"]
                st.session_state["last_scan_result"] = result

                # Save uploaded image and scan record if models are trained
                if result["models_trained"] and result["dsa_result"] is not None:
                    os.makedirs(config.UPLOADS_DIR, exist_ok=True)
                    save_name = f"scan_{int(time.time())}_{uploaded_file.name}"
                    save_path = os.path.join(config.UPLOADS_DIR, save_name)
                    pil_img.save(save_path)

                    ranked_tx = result["dsa_result"]["ranked_treatments"]
                    top_tx_name = (
                        ranked_tx[0]["name"] if ranked_tx else "Consult agronomist"
                    )
                    database.save_scan(
                        user_id=int(user["id"]),
                        user_name=str(user["name"]),
                        user_email=str(user["email"]),
                        crop=str(result["predicted_crop"]),
                        pest_name=str(result["predicted_pest"]),
                        scientific_name=str(result["scientific_name"]),
                        cnn_confidence=float(result["cnn_confidence"]),
                        yolo_confidence=float(result["yolo_confidence"]),
                        affected_area_pct=float(result["affected_area_pct"]),
                        severity=str(result["dsa_result"]["severity"]),
                        top_treatment=top_tx_name,
                        dsa_speed_ms=float(result["dsa_speed_ms"]),
                        image_path=save_path,
                    )
                    st.rerun()

        last_res = st.session_state.get("last_scan_result")
        if last_res is not None:
            if not last_res["models_trained"]:
                st.warning(last_res["status_message"])
                st.info(
                    f"Stage 1 OpenCV 8-element feature array extracted in "
                    f"**{last_res['dsa_speed_ms']:.3f} ms** (`time.perf_counter()`)."
                )
                # Allow interactive DSA pipeline execution on the uploaded image
                with st.expander(
                    "Run 4-Stage DSA Pipeline Directly on Selected Crop/Pest",
                    expanded=True,
                ):
                    pest_names_for_crop = [p["name"] for p in crop_pests]
                    chosen_pest_name = st.selectbox(
                        "Select Pest Class for Hash Table Lookup",
                        options=pest_names_for_crop,
                    )
                    manual_area = st.slider(
                        "Affected Leaf Area % (for Stage 3 Decision Tree)",
                        min_value=1.0,
                        max_value=80.0,
                        value=22.0,
                        step=1.0,
                    )
                    if st.button("Execute & Time 4-Stage DSA Pipeline"):
                        t0 = time.perf_counter()
                        dsa_out = dsa.run_dsa_pipeline(
                            image_bgr=bgr_arr if uploaded_file is not None else None,
                            crop=selected_crop,
                            pest_name=chosen_pest_name,
                            affected_area_pct=manual_area,
                        )
                        measured_ms = round((time.perf_counter() - t0) * 1000.0, 3)
                        st.session_state["last_dsa_speed_ms"] = measured_ms
                        top_tx = (
                            dsa_out["ranked_treatments"][0]["name"]
                            if dsa_out["ranked_treatments"]
                            else "N/A"
                        )
                        database.save_scan(
                            user_id=int(user["id"]),
                            user_name=str(user["name"]),
                            user_email=str(user["email"]),
                            crop=selected_crop,
                            pest_name=chosen_pest_name,
                            scientific_name=(
                                dsa_out["pest_record"]["scientific_name"]
                                if dsa_out["pest_record"]
                                else ""
                            ),
                            cnn_confidence=0.0,
                            yolo_confidence=0.0,
                            affected_area_pct=manual_area,
                            severity=dsa_out["severity"],
                            top_treatment=top_tx,
                            dsa_speed_ms=measured_ms,
                        )
                        _display_dsa_output(st, dsa_out, measured_ms)
            else:
                if last_res["uncertain"]:
                    st.error(last_res["warning_message"])
                if last_res.get("crop_mismatch_note"):
                    st.info(last_res["crop_mismatch_note"])

                st.success(
                    f"Detected Pest: **{last_res['predicted_pest']}** "
                    f"(*{last_res['scientific_name']}*) on **{last_res['predicted_crop']}** "
                    f"| Confidence: **{last_res['fused_confidence'] * 100:.1f}%** "
                    f"| Measured DSA Speed: **{last_res['dsa_speed_ms']:.3f} ms**"
                )

                st.subheader("CNN Top-3 Softmax Predictions (EfficientNet-B0)")
                for rank_idx, pred in enumerate(last_res["cnn_top3"], start=1):
                    st.write(
                        f"**#{rank_idx}** `{pred['crop']}` — **{pred['pest_name']}** "
                        f"(*{pred['scientific_name']}*): `{pred['confidence'] * 100:.2f}%`"
                    )

                if last_res["dsa_result"] is not None:
                    _display_dsa_output(
                        st, last_res["dsa_result"], last_res["dsa_speed_ms"]
                    )

    # Supported Pests Table dynamically titled with NUM_CROPS and NUM_PESTS
    st.markdown("---")
    st.subheader(
        f"Supported IP102 Reference Database ({get_header_scope_text()})"
    )
    table_rows = [
        {
            "Class ID": p["class_id"],
            "Host Crop": p["crop"],
            "Pest Common Name": p["common_name"],
            "Scientific Name": p["scientific_name"],
            "Treatments Available": len(p["treatments"]),
        }
        for p in PEST_DATABASE
    ]
    st.dataframe(table_rows, use_container_width=True, hide_index=True)


def _display_dsa_output(
    st: Any, dsa_result: Dict[str, Any], dsa_speed_ms: float
) -> None:
    """
    Helper to display all 4 DSA stages and ranked treatments on the Scanner page.
    """
    st.markdown(
        f"### 4-Stage DSA Pipeline Results (Measured in `{dsa_speed_ms:.3f} ms`)"
    )
    st.write(
        f"- **Stage 1 (Fixed-Size 8-Element Feature Array):** `{dsa_result['feature_array']}`\n"
        f"- **Stage 2 (31-Bucket Hash Table Lookup):** Key `\"{dsa_result['hash_key']}\"` "
        f"-> Bucket `#{dsa_result['hash_bucket']}` in `{dsa_result['hash_steps']}` chain step(s)\n"
        f"- **Stage 3 (Decision Tree Severity):** **{dsa_result['severity']}** "
        f"(`{dsa_result['decision_path']}`)\n"
        f"- **Strategy:** {dsa_result['severity_strategy']}"
    )

    st.subheader("Stage 4: Priority Queue Max-Heap Ranked Treatments")
    for tx in dsa_result["ranked_treatments"]:
        st.markdown(
            f"**Rank #{tx['rank']} — {tx['name']}** (`{tx['type']}`)  \n"
            f"Priority Score: `{tx['priority_score']}` | Efficacy: `{tx['efficacy']}%` | "
            f"PHI: `{tx['phi_days']} days` | Cost: `Rs {tx['cost_inr']}/acre`  \n"
            f"*Dosage:* {tx['dosage']} — *Safety:* {tx['safety_notes']}"
        )


# # [DATABASE]
def render_history_page(st: Any, user: Dict[str, Any]) -> None:
    """
    Displays the logged-in user's saved scan history and real measured DSA speeds.
    """
    st.header("My Leaf Scan History")
    speed_stats = database.get_dsa_speed_stats()
    st.caption(
        f"Scope: {get_header_scope_text()} | "
        f"DSA Pipeline Speed: {speed_stats['display_text']}"
    )

    scans = database.get_scans(user_id=int(user["id"]))
    if not scans:
        st.info("No scans saved yet. Go to the Leaf Pest Scanner page to run a scan.")
        return

    st.dataframe(scans, use_container_width=True, hide_index=True)


# # [FEEDBACK]
def render_feedback_page(st: Any, user: Dict[str, Any]) -> None:
    """
    Allows the logged-in user to submit complaints/feedback and track status
    and admin response messages right inside their account.
    """
    st.header("My Feedback & Complaint Tracker")
    st.caption(
        "Submit a question or complaint about a scan or treatment recommendation, "
        "and view responses sent by the Admin."
    )

    col_form, col_list = st.columns([1, 1.3])

    with col_form:
        st.subheader("Submit a New Complaint / Feedback")
        with st.form("feedback_form", clear_on_submit=True):
            fb_crop = st.selectbox(
                "Related Crop", options=["General"] + config.CROPS
            )
            fb_pest = st.selectbox(
                "Related Pest",
                options=["General"] + [p["name"] for p in config.PEST_SHORTLIST],
            )
            fb_subject = st.text_input(
                "Complaint Subject",
                value="",
                placeholder="Brief summary of your issue",
            )
            fb_message = st.text_area(
                "Detailed Message",
                value="",
                placeholder="Describe what happened or what guidance you need...",
            )
            fb_submit = st.form_submit_button(
                "Submit Complaint", use_container_width=True
            )

        if fb_submit:
            ok, msg, _ = database.submit_feedback(
                user_id=int(user["id"]),
                user_name=str(user["name"]),
                user_email=str(user["email"]),
                subject=fb_subject,
                message=fb_message,
                crop=fb_crop,
                pest_name=fb_pest,
            )
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    with col_list:
        st.subheader("Track My Submitted Complaints & Admin Responses")
        my_items = database.get_feedback_list(user_id=int(user["id"]))
        if not my_items:
            st.info("You have not submitted any complaints yet.")
        else:
            for item in my_items:
                status_badge = (
                    "RESOLVED" if item["status"] == "Resolved" else "PENDING"
                )
                with st.expander(
                    f"[#{item['id']} - {status_badge}] {item['subject']} "
                    f"(Submitted: {item['created_at']})",
                    expanded=True,
                ):
                    st.write(
                        f"**Submitted by:** {item['user_name']} (`{item['user_email']}`)  \n"
                        f"**Date & Time:** `{item['created_at']}`  \n"
                        f"**Crop / Pest:** `{item['crop']}` / `{item['pest_name']}`  \n"
                        f"**Status:** **{item['status']}**"
                    )
                    st.write(f"**Your Message:** {item['message']}")
                    if item["admin_reply"]:
                        st.success(
                            f"**Admin Response (by {item['replied_by']} at {item['replied_at']}):**  \n"
                            f"{item['admin_reply']}"
                        )
                    else:
                        st.warning(
                            "Awaiting Admin response. Check back here for updates."
                        )


# # [ADMIN] # [FEEDBACK]
def render_admin_page(st: Any, user: Dict[str, Any]) -> None:
    """
    Admin Dashboard:
      - Shows who made each complaint (user's name and email), date and time,
        subject, message, and current status.
      - Lets the Admin send a response message that the user sees in their
        account when they log in.
      - Also shows all registered users, all scans, and model evaluation metrics.
    """
    if user.get("role") != "admin":
        st.error("Access denied. Administrator role required.")
        return

    st.header("Administrator Dashboard")
    st.caption(
        f"Signed in as Admin: {user['name']} ({user['email']}) | "
        f"Scope: {get_header_scope_text()}"
    )

    tab_complaints, tab_scans, tab_users, tab_eval = st.tabs(
        [
            "User Complaints & Reply",
            "All User Scans",
            "Registered Users",
            "Model Evaluation Metrics",
        ]
    )

    with tab_complaints:
        st.subheader("All User Complaints (Who Complained, Date & Time, and Reply)")
        all_fb = database.get_feedback_list(user_id=None)
        if not all_fb:
            st.info("No user complaints have been submitted yet.")
        else:
            for fb in all_fb:
                with st.expander(
                    f"Complaint #{fb['id']} | From: {fb['user_name']} ({fb['user_email']}) "
                    f"| Date/Time: {fb['created_at']} | Status: {fb['status']}",
                    expanded=(fb["status"] != "Resolved"),
                ):
                    st.write(
                        f"- **Complainant Name:** {fb['user_name']}\n"
                        f"- **Complainant Email:** `{fb['user_email']}`\n"
                        f"- **Submitted Date & Time:** `{fb['created_at']}`\n"
                        f"- **Related Crop / Pest:** `{fb['crop']}` / `{fb['pest_name']}`\n"
                        f"- **Subject:** {fb['subject']}\n"
                        f"- **Complaint Message:** {fb['message']}"
                    )
                    if fb["admin_reply"]:
                        st.info(
                            f"**Previous Admin Reply ({fb['replied_by']} at {fb['replied_at']}):** "
                            f"{fb['admin_reply']}"
                        )

                    with st.form(f"admin_reply_form_{fb['id']}"):
                        reply_text = st.text_area(
                            "Send Response Message to User",
                            value=fb["admin_reply"],
                            placeholder="Write your response to this user...",
                        )
                        new_status = st.selectbox(
                            "Update Ticket Status",
                            options=["Resolved", "Pending"],
                            index=0 if fb["status"] == "Resolved" else 1,
                        )
                        send_btn = st.form_submit_button("Send Response to User")

                    if send_btn:
                        ok, msg = database.reply_to_feedback(
                            feedback_id=int(fb["id"]),
                            admin_reply=reply_text,
                            admin_name=str(user["name"]),
                            new_status=new_status,
                        )
                        if ok:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)

    with tab_scans:
        st.subheader("All Saved Scans Across Users")
        all_scans = database.get_scans(user_id=None)
        if not all_scans:
            st.info("No scans recorded yet.")
        else:
            st.dataframe(all_scans, use_container_width=True, hide_index=True)

    with tab_users:
        st.subheader("Registered Users")
        st.dataframe(
            database.get_all_users(), use_container_width=True, hide_index=True
        )

    with tab_eval:
        st.subheader("Real Test-Set Evaluation Metrics (from evaluate.py)")
        eval_row = database.get_latest_evaluation_metrics()
        if eval_row is None:
            st.warning(
                "Not evaluated yet. Train the models and run `python evaluate.py` "
                "to populate real test-set metrics in SQLite."
            )
        else:
            st.json(eval_row)
            if os.path.isfile(config.CONFUSION_MATRIX_PATH):
                st.image(
                    config.CONFUSION_MATRIX_PATH,
                    caption="Saved Test-Set Confusion Matrix",
                )


# # [ARRAY] # [HASH_TABLE] # [DECISION_TREE] # [PRIORITY_QUEUE]
def render_dsa_guide_page(st: Any) -> None:
    """
    Renders the interactive DSA Guide demonstrating all 4 custom data structures
    in `dsa.py` with live inspection and Big-O complexity tables.
    """
    st.header(
        f"Data Structures & Algorithms (DSA) Guide — {get_header_scope_text()}"
    )
    st.write(
        "All 4 classical Data Structures in `dsa.py` are implemented from scratch "
        "in pure Python without built-in `dict` (in the Hash Table) or `heapq` "
        "(in the Priority Queue)."
    )

    st.subheader("Big-O Time & Space Complexity Summary")
    st.table(
        [
            {
                "Stage": "1. Fixed-Size Feature Array (# [ARRAY])",
                "Operation": "Extract 8 OpenCV features & index arr[i]",
                "Time Complexity": "O(H * W) extraction, O(1) index access",
                "Space Complexity": f"O(1) fixed ({config.FEATURE_ARRAY_SIZE} floats)",
            },
            {
                "Stage": f"2. {config.HASH_TABLE_BUCKETS}-Bucket Hash Table (# [HASH_TABLE])",
                "Operation": f"Polynomial Rolling Hash (base {config.HASH_POLYNOMIAL_BASE}) + Separate Chaining",
                "Time Complexity": "O(L) average lookup, O(L + N) worst case",
                "Space Complexity": f"O(B + N) where B={config.HASH_TABLE_BUCKETS}, N={config.NUM_PESTS}",
            },
            {
                "Stage": "3. Severity Decision Tree (# [DECISION_TREE])",
                "Operation": f"Threshold split (<{config.SEVERITY_LOW_MAX}% Low, <= {config.SEVERITY_MODERATE_MAX}% Moderate, >{config.SEVERITY_MODERATE_MAX}% Severe)",
                "Time Complexity": "O(H) = O(1) where tree height H = 2",
                "Space Complexity": "O(1) (5 tree nodes)",
            },
            {
                "Stage": "4. Max-Heap Priority Queue (# [PRIORITY_QUEUE])",
                "Operation": "Manual _sift_up / _sift_down treatment ranking",
                "Time Complexity": "O(T log T) for T treatments",
                "Space Complexity": "O(T)",
            },
        ]
    )

    st.subheader(
        f"Live {config.HASH_TABLE_BUCKETS}-Bucket Hash Table Inspector "
        f"({len(config.PEST_SHORTLIST)} Pests Across {config.NUM_CROPS} Crops)"
    )
    st.dataframe(
        dsa.GLOBAL_HASH_TABLE.get_bucket_summary(),
        use_container_width=True,
        hide_index=True,
    )


# ==============================================================================
# 3. MAIN STREAMLIT ENTRY POINT
# ==============================================================================

def main() -> None:
    """
    Main entry point when running `streamlit run app.py`.
    """
    import streamlit as st

    st.set_page_config(
        page_title=f"Smart Pest Detection ({get_header_scope_text()})",
        layout="wide",
    )

    database.init_db()

    user = st.session_state.get("user")
    if user is None:
        render_login_page(st)
        return

    # Sidebar with dynamic counts from config.py
    with st.sidebar:
        st.title("Smart Pest System")
        st.caption(f"Scope: **{get_header_scope_text()}**")
        st.write(
            f"Signed in: **{user['name']}**  \n"
            f"Role: `{user['role'].upper()}`"
        )
        stats = get_live_dashboard_stats(
            last_scan_speed_ms=st.session_state.get("last_dsa_speed_ms")
        )
        st.markdown("---")
        st.write(f"- **Supported Crops:** {stats['crops_text']}")
        st.write(f"- **Pests Indexed:** {stats['pests_text']}")
        st.write(f"- **Detection Accuracy:** {stats['accuracy_text']}")
        st.write(f"- **DSA Pipeline Speed:** {stats['dsa_speed_text']}")
        st.markdown("---")

        nav_options = [
            "Leaf Pest Scanner",
            "My Scan History",
            "Feedback & Complaints",
            "DSA Guide & Inspector",
        ]
        if user.get("role") == "admin":
            nav_options.append("Admin Dashboard")

        choice = st.radio("Navigation", options=nav_options)

        if st.button("Sign Out", use_container_width=True):
            st.session_state.pop("user", None)
            st.rerun()

    if choice == "Leaf Pest Scanner":
        render_scanner_page(st, user)
    elif choice == "My Scan History":
        render_history_page(st, user)
    elif choice == "Feedback & Complaints":
        render_feedback_page(st, user)
    elif choice == "DSA Guide & Inspector":
        render_dsa_guide_page(st)
    elif choice == "Admin Dashboard":
        render_admin_page(st, user)


if __name__ == "__main__":
    main()
