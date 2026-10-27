"""Reports Page."""
from pathlib import Path
import streamlit as st
from database.repository import ExperimentRepository
from services.report_service import ReportService
from frontend.components import render_section_header, render_empty_state

def render_reports(repo: ExperimentRepository, on_navigate):
    """Generates, visualizes, and exports comprehensive laboratory reports."""
    render_section_header("Laboratory Reports Archive", "Compile formal laboratory experiment documentation in Markdown, HTML, or structured JSON.")

    curr_id = st.session_state.get("current_experiment_id")
    experiments = repo.list_experiments(limit=100)

    if not experiments:
        render_empty_state("No Experiments Available", "Create experiments and run models to compile scientific reports.")
        return

    exp_options = {f"{e.name} (ID: {e.id[:8]})": e.id for e in experiments}
    default_idx = 0
    if curr_id:
        for idx, (lbl, eid) in enumerate(exp_options.items()):
            if eid == curr_id:
                default_idx = idx
                break

    selected_label = st.selectbox("Select Experiment for Report Generation:", list(exp_options.keys()), index=default_idx)
    target_exp_id = exp_options[selected_label]
    target_exp = repo.get_experiment(target_exp_id)

    c_fmt, c_btn = st.columns([2, 1])
    with c_fmt:
        report_format = st.selectbox("Report Format:", ["markdown", "html", "json"], help="Markdown is ideal for reading and GitHub documentation; HTML provides self-contained styled styling; JSON exports structured metadata.")
    with c_btn:
        st.markdown("<div style='padding-top: 28px;'></div>", unsafe_allow_html=True)
        generate_btn = st.button("📄 Generate New Report", type="primary", use_container_width=True)

    report_service = ReportService(repo=repo)

    if generate_btn:
        with st.spinner(f"Compiling {report_format.upper()} report from saved records..."):
            rep_path = report_service.generate_report(target_exp_id, report_format=report_format)
            st.session_state["last_generated_report"] = str(rep_path)
            st.success(f"Generated report: `{rep_path.name}`")

    # Display report preview and download button
    last_rep_path = st.session_state.get("last_generated_report")
    if last_rep_path and Path(last_rep_path).exists():
        p = Path(last_rep_path)
        with open(p, "rb") as f:
            file_bytes = f.read()

        mime_type = "text/markdown" if p.suffix == ".md" else ("text/html" if p.suffix == ".html" else "application/json")
        st.download_button(
            label=f"📥 Download Report ({p.name})",
            data=file_bytes,
            file_name=p.name,
            mime=mime_type,
            type="primary"
        )

        st.markdown("#### Document Preview")
        if p.suffix == ".md":
            st.markdown(file_bytes.decode("utf-8"))
        elif p.suffix == ".html":
            st.components.v1.html(file_bytes.decode("utf-8"), height=600, scrolling=True)
        else:
            st.json(file_bytes.decode("utf-8"))

    # Display Previously Compiled Reports
    st.markdown("---")
    st.markdown("#### Saved Reports History")
    if target_exp and target_exp.reports:
        for r in reversed(target_exp.reports):
            r_p = Path(r.file_path)
            if r_p.exists():
                c1, c2, c3 = st.columns([3, 2, 1.5])
                with c1:
                    st.markdown(f"**{r.title}**")
                with c2:
                    st.markdown(f"Format: `.{r.report_type}` | {r.created_at.strftime('%Y-%m-%d %H:%M')}")
                with c3:
                    with open(r_p, "rb") as f:
                        data_b = f.read()
                    st.download_button("Download", data=data_b, file_name=r_p.name, key=f"dl_rep_{r.id}", use_container_width=True)
    else:
        st.info("No previously generated reports stored for this experiment yet.")
