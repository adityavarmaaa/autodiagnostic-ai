import streamlit as st

from backend.diagnostic import DiagnosticService
from backend.rag import RAGService


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AutoDiag AI",
    page_icon="🚗",
    layout="wide",
)


# =========================================================
# SERVICES
# =========================================================

@st.cache_resource
def get_diagnostic_service():
    """
    Create the diagnostic service once and reuse it.
    """
    return DiagnosticService()


@st.cache_resource
def get_rag_service():
    """
    Create RAG service and automatically synchronize
    documents from data/manuals/.

    Important:
    Knowledge already stored in ChromaDB is persistent.
    Deleting the original PDF/TXT does not delete its
    stored knowledge.
    """
    rag = RAGService()

    sync_result = rag.sync_local_documents()

    return rag, sync_result


diagnostic_service = get_diagnostic_service()
rag_service, kb_sync = get_rag_service()


# =========================================================
# HEADER
# =========================================================

st.title("🚗 AutoDiag AI")

st.caption(
    "Automotive Diagnostic Copilot powered by "
    "RAG, ChromaDB, Ollama and web intelligence."
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📚 Knowledge Base")

    st.metric(
        "Indexed Knowledge Chunks",
        rag_service.count(),
    )

    st.divider()

    # -----------------------------------------------------
    # Knowledge Base Sync Status
    # -----------------------------------------------------

    st.subheader("Knowledge Base Status")

    added = kb_sync.get(
        "added",
        [],
    )

    changed = kb_sync.get(
        "changed",
        [],
    )

    skipped = kb_sync.get(
        "skipped",
        [],
    )

    failed = kb_sync.get(
        "failed",
        [],
    )

    # New documents
    if added:

        st.success(
            f"📥 {len(added)} new document(s) indexed."
        )

        for item in added:

            st.write(
                f"• {item['source']} "
                f"({item['chunks']} chunks)"
            )

    # Changed documents
    if changed:

        st.info(
            f"🔄 {len(changed)} document(s) updated."
        )

        for item in changed:

            st.write(
                f"• {item['source']} "
                f"({item['chunks']} chunks)"
            )

    # Unchanged documents
    if skipped:

        st.caption(
            f"✓ {len(skipped)} document(s) "
            f"already up to date."
        )

    # Failed documents
    if failed:

        st.error(
            f"⚠️ {len(failed)} document(s) failed."
        )

        for item in failed:

            st.caption(
                f"• {item.get('source', 'Unknown')}: "
                f"{item.get('error', 'Unknown error')}"
            )

    st.divider()

    # -----------------------------------------------------
    # System
    # -----------------------------------------------------

    st.subheader("⚙️ System")

    st.write("✅ Persistent Knowledge Base")
    st.write("✅ Local RAG")
    st.write("✅ ChromaDB")
    st.write("✅ Ollama AI")
    st.write("✅ Web intelligence")
    st.write("✅ Vehicle relevance filtering")

    st.divider()

    st.info(
        "AutoDiag searches stored knowledge first. "
        "If relevant knowledge is unavailable, "
        "it can use web research."
    )


# =========================================================
# DIAGNOSTIC CASE
# =========================================================

st.header("🔧 Diagnostic Case")

st.write(
    "Enter the vehicle information and customer complaint."
)


# =========================================================
# VEHICLE
# =========================================================

vehicle_col, year_col = st.columns(2)

with vehicle_col:

    vehicle = st.text_input(
        "Vehicle",
        placeholder="Mercedes-Benz AMG G63",
    )

with year_col:

    year = st.text_input(
        "Year",
        placeholder="2026",
    )


# =========================================================
# COMPLAINT
# =========================================================

complaint = st.text_area(
    "Customer Complaint",
    placeholder=(
        "Example: Engine shaking when stopped at idle"
    ),
    height=130,
)


# =========================================================
# DTC
# =========================================================

dtc = st.text_input(
    "DTC Code",
    placeholder="P0301",
    help="Optional. Example: P0301, P0171, U0121",
)


st.write("")


# =========================================================
# ANALYZE BUTTON
# =========================================================

analyze = st.button(
    "🔍 Analyze Complaint",
    type="primary",
    use_container_width=True,
)


# =========================================================
# ANALYSIS
# =========================================================

if analyze:

    # -----------------------------------------------------
    # Validate input
    # -----------------------------------------------------

    if not vehicle.strip():

        st.error(
            "Please enter the vehicle."
        )

        st.stop()

    if not complaint.strip():

        st.error(
            "Please enter the customer complaint."
        )

        st.stop()

    # -----------------------------------------------------
    # Build complete vehicle identity
    # -----------------------------------------------------

    full_vehicle = vehicle.strip()

    if year.strip():

        full_vehicle += (
            f" {year.strip()}"
        )

    # -----------------------------------------------------
    # Run diagnostic pipeline
    # -----------------------------------------------------

    with st.spinner(
        "🔍 Retrieving technical information "
        "and generating diagnostic report..."
    ):

        try:

            result = diagnostic_service.analyze(
                vehicle=full_vehicle,
                complaint=complaint.strip(),
                dtc=dtc.strip(),
                top_k=3,
            )

        except Exception as exc:

            st.error(
                "Unable to analyze the case."
            )

            st.exception(exc)

            st.stop()


    # =====================================================
    # DIAGNOSTIC REPORT
    # =====================================================

    st.divider()

    st.header("🤖 AI Diagnostic Report")

    answer = result.get(
        "answer",
        "",
    )

    if answer:

        st.markdown(
            answer
        )

    else:

        st.warning(
            "The AI did not return a diagnostic report."
        )


    # =====================================================
    # CASE SUMMARY
    # =====================================================

    st.divider()

    st.subheader("📋 Case Summary")

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:

        st.write(
            f"**Vehicle:** {full_vehicle}"
        )

        st.write(
            f"**Complaint:** {complaint.strip()}"
        )

    with summary_col2:

        st.write(
            f"**DTC:** "
            f"{dtc.strip() or 'Not provided'}"
        )

        st.write(
            f"**Search Query:** "
            f"{result.get('search_query', 'N/A')}"
        )


    # =====================================================
    # SAFETY
    # =====================================================

    st.warning(
        """
        ⚠️ Important:

        This report provides possible diagnostic guidance,
        not a confirmed diagnosis.

        Verify findings using the applicable
        manufacturer procedures before repairing
        or replacing components.
        """
    )


    # =====================================================
    # RETRIEVED TECHNICAL SOURCES
    # =====================================================

    st.divider()

    st.header("📚 Retrieved Technical Sources")

    sources = result.get(
        "sources",
        [],
    )

    if not sources:

        st.info(
            "No relevant technical sources were retrieved."
        )

    else:

        st.write(
            f"Retrieved {len(sources)} relevant source(s)."
        )

        for index, source in enumerate(
            sources,
            start=1,
        ):

            source_type = source.get(
                "type",
                "unknown",
            )

            source_name = source.get(
                "source",
                "Unknown source",
            )

            page = source.get(
                "page",
                "",
            )

            chunk_index = source.get(
                "chunk_index",
                "?",
            )

            # -------------------------------------------------
            # Source title
            # -------------------------------------------------

            if source_type == "pdf":

                source_title = (
                    f"{index}. 📘 {source_name} "
                    f"— Page {page}"
                )

            elif source_type == "txt":

                source_title = (
                    f"{index}. 📄 {source_name} "
                    f"— Local Knowledge Base"
                )

            elif source_type == "web":

                source_title = (
                    f"{index}. 🌐 {source_name} "
                    f"— Web Source"
                )

            else:

                source_title = (
                    f"{index}. 📎 {source_name}"
                )

            # -------------------------------------------------
            # Expand source
            # -------------------------------------------------

            with st.expander(
                source_title
            ):

                # =============================================
                # WEB URL
                # =============================================

                source_url = source.get(
                    "url",
                    "",
                )

                if source_url:

                    st.markdown(
                        f"🔗 **[Open original source]({source_url})**"
                    )

                    st.caption(
                        source_url
                    )

                # =============================================
                # SOURCE METADATA
                # =============================================

                if source_type == "web":

                    st.caption(
                        f"Web chunk #{chunk_index}"
                    )

                elif source_type == "pdf":

                    st.caption(
                        f"PDF page {page}, "
                        f"chunk {chunk_index}"
                    )

                elif source_type == "txt":

                    st.caption(
                        f"Local document, "
                        f"chunk {chunk_index}"
                    )

                # =============================================
                # EXTRACTED TEXT
                # =============================================

                st.markdown(
                    "**Extracted technical information:**"
                )

                st.write(
                    source.get(
                        "text",
                        "",
                    )
                )

                # =============================================
                # RETRIEVAL DISTANCE
                # =============================================

                distance = source.get(
                    "distance"
                )

                if distance is not None:

                    st.caption(
                        f"Retrieval distance: "
                        f"{distance:.4f}"
                    )


    # =====================================================
    # WEB RESEARCH STATUS
    # =====================================================

    if result.get(
        "used_web_search",
        False,
    ):

        st.info(
            "🌐 Local technical evidence was insufficient, "
            "so web research was used."
        )


    # =====================================================
    # WEB SOURCES
    # =====================================================

    web_sources = result.get(
        "web_sources",
        [],
    )

    if web_sources:

        st.divider()

        st.subheader(
            "🌐 Web Sources Used"
        )

        for index, source in enumerate(
            web_sources,
            start=1,
        ):

            # -------------------------------------------------
            # New format:
            # {
            #     "title": "...",
            #     "url": "..."
            # }
            # -------------------------------------------------

            if isinstance(
                source,
                dict,
            ):

                title = source.get(
                    "title",
                    "Web source",
                )

                url = source.get(
                    "url",
                    "",
                )

                st.markdown(
                    f"**{index}. {title}**"
                )

                if url:

                    st.markdown(
                        f"🔗 [Open source]({url})"
                    )

                    st.caption(
                        url
                    )

                st.write("")

            # -------------------------------------------------
            # Backward compatibility:
            # old format was simply a URL string
            # -------------------------------------------------

            else:

                source_text = str(
                    source
                )

                st.markdown(
                    f"**{index}.**"
                )

                if source_text.startswith(
                    "http://"
                ) or source_text.startswith(
                    "https://"
                ):

                    st.markdown(
                        f"🔗 [Open source]({source_text})"
                    )

                    st.caption(
                        source_text
                    )

                else:

                    st.write(
                        source_text
                    )


    # =====================================================
    # KNOWLEDGE STATUS
    # =====================================================

    st.divider()

    st.subheader(
        "🧠 Knowledge Status"
    )

    st.write(
        "The response can use information retrieved "
        "from the persistent AutoDiag Knowledge Base."
    )

    if result.get(
        "used_web_search",
        False,
    ):

        st.caption(
            "This case also required fresh web research."
        )

    else:

        st.caption(
            "This case was answered using available "
            "stored knowledge."
        )


    # =====================================================
    # FEEDBACK
    # =====================================================

    st.divider()

    st.subheader(
        "Was this diagnostic guidance useful?"
    )

    feedback_col1, feedback_col2 = st.columns(2)

    with feedback_col1:

        if st.button(
            "👍 Useful",
            use_container_width=True,
        ):

            st.success(
                "Thanks for the feedback!"
            )

    with feedback_col2:

        if st.button(
            "👎 Not useful",
            use_container_width=True,
        ):

            st.info(
                "Thanks. This helps improve AutoDiag AI."
            )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "AutoDiag AI • Evidence-grounded automotive "
    "diagnostic assistance • Verify all repairs "
    "with qualified procedures."
)