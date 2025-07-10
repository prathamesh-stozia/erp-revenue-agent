import streamlit as st
import pandas as pd
import asyncio
import os
from datetime import datetime
from pathlib import Path
import requests
from markdownify import markdownify
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import urllib3
from typing import Optional, List, Dict
import json

# Code from agent.py (excluding if __name__ == "__main__": block)
load_dotenv()
SERPER_API_KEY = None
OPENAI_API_KEY = None

openai_client = None
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class ERPRevenueInfo(BaseModel):
    erp_system: Optional[str] = Field(description="The ERP system used by the company")
    erp_confidence: int = Field(description="Confidence level in the ERP system")
    annual_revenue: Optional[str] = Field(description="Annual revenue")
    revenue_confidence: int = Field(description="Confidence level in revenue")
    revenue_range: Optional[str] = Field(description="Revenue range")
    company_size: Optional[str] = Field(description="Company size")
    business_type: Optional[str] = Field(description="Business type")
    notes: Optional[str] = Field(description="Additional notes")

class LinkScore(BaseModel):
    url: str = Field(description="The URL")
    score: int = Field(description="Score from 1-10")
    reason: str = Field(description="Reason for score")

class CuratedLinks(BaseModel):
    scored_links: List[LinkScore] = Field(description="List of scored links")

async def web_search(query: str, serper_key: str) -> list[dict]:
    """Performs a web search using Serper API and returns a list of results."""
    print(f"\t🔎 Searching for: '{query}'")
    try:
        url = "https://google.serper.dev/search"
        
        payload = json.dumps({
            "q": query,
            "gl": "in",  
            "num": 8     
        })
        
        headers = {
            'X-API-KEY': serper_key,
            'Content-Type': 'application/json'
        }
        
        response = requests.post(url, headers=headers, data=payload, timeout=10)
        response.raise_for_status()
        
        search_results = response.json()
        
        formatted_results = []
        for result in search_results.get('organic', []):
            formatted_results.append({
                "title": result.get("title", "No Title"),
                "url": result.get("link", ""),
                "snippet": result.get("snippet", "No description available")
            })
        return formatted_results
    except Exception as e:
        print(f"\t❌ Error during search for '{query}': {e}")
        return []

async def read_webpage_text(url: str, use_selenium: bool = False) -> str:
    """Reads a webpage, converts its HTML to Markdown, and returns the text."""
    print(f"\t📄 Reading: {url}")
    
    if not use_selenium:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            response = requests.get(
                url, 
                headers=headers, 
                timeout=15,
                verify=False,
                allow_redirects=True
            )
            response.raise_for_status()
            
            markdown_text = markdownify(response.text)
            return markdown_text
            
        except Exception as e:
            print(f"\t⚠️ Requests failed for '{url}': {e}")
            return ""
    
    return ""

async def curate_links(links: List[dict], company_name: str, search_type: str, openai_key: str) -> List[dict]:
    """Uses LLM to score and curate the best links based on search type (revenue or ERP)."""
    print(f"\t🎯 Curating links for {search_type} research...")
    
    if not links:
        return []
    
    link_info = []
    for link in links:
        url = link.get('url', '')
        title = link.get('title', 'No Title')
        snippet = link.get('snippet', 'No description available')
        link_info.append(f"URL: {url}\nTitle: {title}\nSnippet: {snippet}")
    
    links_text = "\n\n".join(link_info)
    
    if search_type.lower() == "revenue":
        prompt = f"""You are a research analyst evaluating web links for finding REVENUE and FINANCIAL information about '{company_name}'.

Score each link from 1-10 based on:\n- Likelihood of containing revenue/financial data (70%)\n- Source credibility and reliability (20%)\n- Accessibility (no paywalls/login required) (10%)\n
PRIORITIZE for REVENUE search:\n- Company annual reports and financial statements\n- SEC filings and investor relations pages\n- Business directories with revenue/turnover data\n- Financial news articles with specific revenue figures\n- Government databases with financial information\n- Company websites with investor/financial sections\n- Industry reports with company financials\n
AVOID for REVENUE search:\n- Job posting sites\n- General company descriptions without financial data\n- Social media posts\n- News articles without specific financial figures\n- Technology/product focused content\n
Links to evaluate:\n{links_text}\n
Focus specifically on revenue, turnover, financial performance, and company size indicators."""

    else:  # ERP search
        prompt = f"""You are a research analyst evaluating web links for finding ERP SYSTEMS and TECHNOLOGY information about '{company_name}'.

Score each link from 1-10 based on:\n- Likelihood of containing ERP/technology system information (70%)\n- Source credibility and reliability (20%)\n- Accessibility (no paywalls/login required) (10%)\n
PRIORITIZE for ERP search:\n- News articles about technology implementations\n- Company websites mentioning software systems\n- Job postings mentioning specific ERP systems\n- Case studies and success stories\n- Technology vendor websites with customer lists\n- Industry publications about tech adoption\n- Company press releases about system implementations\n- IT consulting firm websites with client mentions\n- Business directories with technology details\n
AVOID for ERP search:\n- Pure financial/revenue focused content\n- General company descriptions without tech details\n- Job postings (unless they mention specific ERP systems)\n- Social media posts\n- Unrelated technology news\n
Look specifically for mentions of (if possible): SAP, Oracle, Microsoft Dynamics, Salesforce, NetSuite, Tally, QuickBooks, ERP systems, business software, enterprise systems.\n
Links to evaluate:\n{links_text}\n
Focus specifically on ERP systems, business software, and technology implementations."""

    global openai_client
    openai_client = AsyncOpenAI(api_key=openai_key)
    
    try:
        curated = await llm_response_structured_object(prompt, CuratedLinks, openai_key)
        scored_links = sorted(curated.scored_links, key=lambda x: x.score, reverse=True)
        
        curated_links = []
        for scored_link in scored_links:
            if scored_link.score >= 6:
                for original_link in links:
                    if original_link.get('url') == scored_link.url:
                        curated_links.append(original_link)
                        break
        
        return curated_links[:3]  # Return top 3 curated links
        
    except Exception as e:
        print(f"\t❌ Error during {search_type} curation: {e}")
        return links[:3]

async def llm_response_structured_object(prompt: str, schema: BaseModel, openai_key: str) -> BaseModel:
    """Gets a structured JSON response from an LLM based on a schema."""
    openai_client = AsyncOpenAI(api_key=openai_key)
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            tools=[{"type": "function", "function": {"name": schema.__name__, "description": schema.__doc__, "parameters": schema.model_json_schema()}}],
            tool_choice={"type": "function", "function": {"name": schema.__name__}},
            temperature=0
        )
        arguments = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        return schema(**arguments)
    except Exception as e:
        print(f"\t❌ Error getting structured LLM response: {e}")
        return schema()

async def extract_erp_revenue_info(content: str, company_name: str, openai_key: str) -> ERPRevenueInfo:
    """Extract ERP and revenue information from scraped content using LLM."""
    prompt = f"""You are a business analyst extracting ERP system and revenue information from web content about '{company_name}'.

Analyze the following content and extract:\n1. ERP SYSTEM: Look for mentions of:\n   - Enterprise Resource Planning (ERP) systems\n   - Business management software\n   - Specific ERP brands: SAP, Oracle, Microsoft Dynamics, Salesforce, NetSuite, Tally, QuickBooks, etc.\n   - Implementation partners or consultants\n   - System integrations\n\n2. REVENUE INFORMATION: Look for:\n   - Annual revenue/turnover figures\n   - Sales figures\n   - Financial performance data\n   - Revenue growth percentages\n   - Market capitalization\n\n3. COMPANY SIZE INDICATORS: Look for:\n   - Company classification (SME, Large Enterprise, etc.)\n   - Business scale indicators\n\nCONTENT TO ANALYZE:\n{content}\n
Instructions:\n- Be conservative with confidence scores\n- For revenue, include currency and year if available\n- Categorize revenue into ranges\n- If no clear information is found, set fields to null and confidence to 1-3\n- Provide reasoning in notes field"""
    try:
        result = await llm_response_structured_object(prompt, ERPRevenueInfo, openai_key)
        return result
    except Exception as e:
        print(f"\t❌ Error extracting ERP/revenue info: {e}")
        return ERPRevenueInfo()

def deduplicate_links(links: List[dict]) -> List[dict]:
    """Remove duplicate URLs from a list of link dictionaries."""
    seen_urls = set()
    deduplicated = []
    
    for link in links:
        url = link.get('url', '')
        if url and url not in seen_urls:
            seen_urls.add(url)
            deduplicated.append(link)
    
    return deduplicated

async def research_company_erp_revenue(company_name: str, location: str = "", industry: str = "", serper_key: str = "", openai_key: str = "") -> Dict:
    """Research a single company for ERP and revenue information."""
    print(f"\n🔍 Researching: {company_name}")
    
    search_context = company_name
    if location:
        search_context += f" {location}"
    if industry:
        search_context += f" {industry}"
    
    search_queries = [
        (f"{company_name} revenue", "revenue"),
        (f"{company_name} \"ERP\"", "ERP")
    ]
    
    all_curated_links = []
    
    for query, search_type in search_queries:
        print(f"\t🔎 Processing {search_type} query: {query}")
        results = await web_search(query, serper_key)
        
        if results:
            curated_for_query = await curate_links(results, company_name, search_type, openai_key)
            all_curated_links.extend(curated_for_query)
    
    curated_links = deduplicate_links(all_curated_links)
    
    all_content = []
    for link in curated_links:
        url = link.get('url')
        if url:
            content = await read_webpage_text(url)
            if content:
                all_content.append(content)
    
    combined_content = "\n\n---\n\n".join(all_content)
    
    if combined_content.strip():
        erp_revenue_info = await extract_erp_revenue_info(combined_content, company_name, openai_key)
    else:
        erp_revenue_info = ERPRevenueInfo(
            erp_system=None,
            erp_confidence=1,
            annual_revenue=None,
            revenue_confidence=1,
            revenue_range=None,
            company_size=None,
            business_type=None,
            notes="No content could be scraped from search results"
        )
    
    return {
        "company_name": company_name,
        "location": location,
        "industry": industry,
        "erp_system": erp_revenue_info.erp_system,
        "erp_confidence": erp_revenue_info.erp_confidence,
        "annual_revenue": erp_revenue_info.annual_revenue,
        "revenue_confidence": erp_revenue_info.revenue_confidence,
        "revenue_range": erp_revenue_info.revenue_range,
        "company_size": erp_revenue_info.company_size,
        "business_type": erp_revenue_info.business_type,
        "notes": erp_revenue_info.notes,
        "sources_found": len(curated_links)
    }

# Code from main.py
st.set_page_config(page_title="ERP & Revenue Research Agent", page_icon="🏢", layout="wide")

st.title("🏢 ERP & Revenue Research Agent")

with st.sidebar:
    st.header("API Keys")
    serper_key = st.text_input("Serper API Key", type="password")
    openai_key = st.text_input("OpenAI API Key", type="password")
    if serper_key and openai_key:
        st.success("API keys provided")
    else:
        st.warning("Enter both API keys")

uploaded_file = st.file_uploader("Upload CSV or Excel file with company names", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df_input = pd.read_csv(uploaded_file)
    else:
        df_input = pd.read_excel(uploaded_file)
    st.write("Preview:")
    st.dataframe(df_input.head())

    cols = list(df_input.columns)
    company_col = st.selectbox("Company name column", cols, index=cols.index("company_name") if "company_name" in cols else 0)
    loc_col = st.selectbox("Location column (optional)", ["None"] + cols)
    ind_col = st.selectbox("Industry column (optional)", ["None"] + cols)

    if st.button("Run Research"):
        if not (serper_key and openai_key):
            st.error("Please set API keys")
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
                    result = asyncio.run(research_company_erp_revenue(company, location, industry, serper_key, openai_key))
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

            csv_data = results_df.to_csv(index=False).encode("utf-8")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_name = f"erp_revenue_results_{ts}.csv"
            st.download_button("Download CSV", csv_data, file_name=default_name, mime="text/csv")
else:
    st.info("Upload a CSV or Excel file to begin") 