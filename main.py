import streamlit as st
import pandas as pd
import asyncio
import os
from datetime import datetime
from pathlib import Path

from agent import research_company_erp_revenue  # reuse existing async function

st.set_page_config(page_title="ERP & Revenue Research Agent", page_icon="🏢", layout="wide")

st.title("🏢 ERP & Revenue Research Agent")

# Sidebar for API keys
with st.sidebar:
    st.header("API Keys")
    serper_key = st.text_input("Serper API Key", type="password")
    openai_key = st.text_input("OpenAI API Key", type="password")
    if serper_key and openai_key:
        os.environ["SERPER_API_KEY"] = serper_key
        os.environ["OPENAI_API_KEY"] = openai_key
        st.success("API keys set")
    else:
        st.warning("Enter both API keys")

# File upload
uploaded_file = st.file_uploader("Upload CSV or Excel file with company names", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df_input = pd.read_csv(uploaded_file)
    else:
        df_input = pd.read_excel(uploaded_file)
    st.write("Preview:")
    st.dataframe(df_input.head())

    # Column selection
    cols = list(df_input.columns)
    company_col = st.selectbox("Company name column", cols, index=cols.index("company_name") if "company_name" in cols else 0)
    loc_col = st.selectbox("Location column (optional)", ["None"] + cols)
    ind_col = st.selectbox("Industry column (optional)", ["None"] + cols)

    # Run button
    if st.button("Run Research"):
        if not (serper_key and openai_key):
            st.error("Please set API keys in the sidebar")
        else:
            total = len(df_input)
            progress_bar = st.progress(0)
            status = st.empty()
            results = []

            for idx, row in df_input.iterrows():
                company = str(row[company_col]).strip()
                location = str(row[loc_col]).strip() if loc_col != "None" and pd.notna(row[loc_col]) else ""
                industry = str(row[ind_col]).strip() if ind_col != "None" and pd.notna(row[ind_col]) else ""

                status.text(f"Processing {idx + 1}/{total}: {company}")
                try:
                    result = asyncio.run(research_company_erp_revenue(company, location, industry))
                    results.append(result)
                except Exception as e:
                    results.append({
                        "company_name": company,
                        "location": location,
                        "industry": industry,
                        "erp_system": None,
                        "erp_confidence": 0,
                        "annual_revenue": None,
                        "revenue_confidence": 0,
                        "revenue_range": None,
                        "company_size": None,
                        "business_type": None,
                        "notes": f"Error: {e}",
                        "sources_found": 0,
                    })

                progress_bar.progress((idx + 1) / total)

            status.text("Done")
            results_df = pd.DataFrame(results)
            st.subheader("Results")
            st.dataframe(results_df)

            # Download
            csv_data = results_df.to_csv(index=False).encode("utf-8")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_name = f"erp_revenue_results_{ts}.csv"
            st.download_button("Download CSV", csv_data, file_name=default_name, mime="text/csv")
else:
    st.info("Upload a CSV or Excel file to begin")