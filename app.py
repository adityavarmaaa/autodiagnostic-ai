import streamlit as st

from backend.diagnostic import DiagnosticService
from backend.rag import RAGService


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="AutoDiag AI",
    page_icon="🚗",
    layout="wide",
)


# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 25px;
    }

    .warning-box {
        padding: 15px;
        border-radius: 8px;
        background-color: #fff3cd;
        border: 1px solid #ffecb5;
        color: #664d03;
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Services
# ---------------------------------------------------------

@st.cache_resource
def get_diagnostic_service():
    return DiagnosticService()


@st.cache_resource
def get_rag_service():
    return RAGService()


diagnostic_service = get_diagnostic_service()
rag_service = get_rag_service()


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">🚗 AutoDiag AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Automotive Diagnostic Copilot'
    '</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:

    st.header("Knowledge Base")

    chunk_count = rag_service.count()

    st.metric(
        "Indexed document chunks",
        chunk_count,
    )

    st.divider()

    st.markdown(
        """
        **How AutoDiag AI works**

        1. Vehicle complaint is entered.
        2. Relevant technical information is retrieved.
        3. Ollama analyzes the retrieved information.
        4. A structured diagnostic report is generated.

        ⚠️ AI suggestions must be verified by a
        qualified technician before repair.
        """
    )


# ---------------------------------------------------------
# Input section
# ---------------------------------------------------------

st.header("Diagnostic Case")

col1, col2 = st.columns(2)

with col1:

    vehicle = st.text_input(
        "Vehicle",
        placeholder="Hyundai i20",
    )

with col2:

    year = st.text_input(
        "Year",
        placeholder="2019",
    )


complaint = st.text_area(
    "Customer Complaint",
    placeholder="Engine shaking when stopped at idle",
    height=120,
)


dtc = st.text_input(
    "DTC Code",
    placeholder="P0301",
)


# ---------------------------------------------------------
# Analyze
# ---------------------------------------------------------

analyze = st.button(
    "🔍 Analyze Complaint",
    type="primary",
    use_container_width=True,
)


if analyze:

    if not vehicle.strip():

        st.error(
            "Please enter the vehicle."
        )

    elif not complaint.strip():

        st.error(
            "Please enter the customer complaint."
        )

    else:

        full_vehicle = vehicle.strip()

        if year.strip():
            full_vehicle += (
                f" {year.strip()}"
            )

        with st.spinner(
            "Retrieving technical information and "
            "generating diagnostic report..."
        ):

            try:

                result = diagnostic_service.analyze(
                    vehicle=full_vehicle,
                    complaint=complaint,
                    dtc=dtc,
                    top_k=5,
                )

            except Exception as exc:

                st.error(
                    "Unable to analyze the case."
                )

                st.exception(exc)

                st.stop()


        # -------------------------------------------------
        # Results
        # -------------------------------------------------

        st.divider()

        st.header(
            "AI Diagnostic Report"
        )

        st.markdown(
            result["answer"]
        )


        # -------------------------------------------------
        # Safety notice
        # -------------------------------------------------

        st.markdown(
            """
            <div class="warning-box">
            ⚠️ <strong>Important:</strong>
            This report provides possible diagnostic guidance,
            not a confirmed diagnosis. Verify findings using
            the applicable vehicle manufacturer's procedures
            before repairing or replacing components.
            </div>
            """,
            unsafe_allow_html=True,
        )


        # -------------------------------------------------
        # Sources
        # -------------------------------------------------

        st.divider()

        st.subheader(
            "📚 Retrieved Technical Sources"
        )

        sources = result["sources"]

        if not sources:

            st.info(
                "No technical documents were retrieved."
            )

        else:

            for index, source in enumerate(
                sources,
                start=1,
            ):

                with st.expander(
                    f"{index}. "
                    f"{source['source']} "
                    f"— Page {source['page']}"
                ):

                    st.write(
                        source["text"]
                    )


        # -------------------------------------------------
        # Feedback
        # -------------------------------------------------

        st.divider()

        st.subheader(
            "Was this diagnostic guidance useful?"
        )

        feedback_col1, feedback_col2 = (
            st.columns(2)
        )

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
                    "Thanks. This helps us improve AutoDiag AI."
                )