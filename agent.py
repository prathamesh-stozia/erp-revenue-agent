import os
import sys
import json
import asyncio
import pandas as pd
from typing import List, Optional, Dict
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

# --- Setup: Load API Keys ---
load_dotenv()
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SERPER_API_KEY or not OPENAI_API_KEY:
    print("Error: SERPER_API_KEY and OPENAI_API_KEY must be set in your environment or a .env file.")
    sys.exit(1)

# Initialize API clients
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Disable SSL warnings for problematic sites
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- Pydantic Models ---

class ERPRevenueInfo(BaseModel):
    """Model for ERP and revenue information."""
    erp_system: Optional[str] = Field(description="The ERP system used by the company (e.g., SAP, Oracle, Microsoft Dynamics, Tally, etc.)")
    erp_confidence: int = Field(description="Confidence level (1-10) in the ERP system identification")
    annual_revenue: Optional[str] = Field(description="Annual revenue of the company (with currency and year if available)")
    revenue_confidence: int = Field(description="Confidence level (1-10) in the revenue information")
    revenue_range: Optional[str] = Field(description="Revenue range category (e.g., 'Under $1M', '$1M-$10M', '$10M-$50M', '$50M-$100M', '$100M+')")
    company_size: Optional[str] = Field(description="Company size category (Startup, Small, Medium, Large, Enterprise)")
    business_type: Optional[str] = Field(description="Type of business (Manufacturer, Distributor, Service Provider, etc.)")
    notes: Optional[str] = Field(description="Additional notes about ERP/revenue findings")



class LinkScore(BaseModel):
    """A model for scoring individual links."""
    url: str = Field(description="The URL being scored")
    score: int = Field(description="Score from 1-10 based on relevance and quality")
    reason: str = Field(description="Brief explanation for the score")

class CuratedLinks(BaseModel):
    """A model for curated and scored links."""
    scored_links: List[LinkScore] = Field(description="List of links with scores and reasoning")

# --- Core Functions (Adapted from research_script_v2.py) ---

async def web_search(query: str) -> list[dict]:
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
            'X-API-KEY': SERPER_API_KEY,
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

async def curate_links(links: List[dict], company_name: str, search_type: str) -> List[dict]:
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

Score each link from 1-10 based on:
- Likelihood of containing revenue/financial data (70%)
- Source credibility and reliability (20%)
- Accessibility (no paywalls/login required) (10%)

PRIORITIZE for REVENUE search:
- Company annual reports and financial statements
- SEC filings and investor relations pages
- Business directories with revenue/turnover data
- Financial news articles with specific revenue figures
- Government databases with financial information
- Company websites with investor/financial sections
- Industry reports with company financials

AVOID for REVENUE search:
- Job posting sites
- General company descriptions without financial data
- Social media posts
- News articles without specific financial figures
- Technology/product focused content

Links to evaluate:
{links_text}

Focus specifically on revenue, turnover, financial performance, and company size indicators."""

    else:  # ERP search
        prompt = f"""You are a research analyst evaluating web links for finding ERP SYSTEMS and TECHNOLOGY information about '{company_name}'.

Score each link from 1-10 based on:
- Likelihood of containing ERP/technology system information (70%)
- Source credibility and reliability (20%)
- Accessibility (no paywalls/login required) (10%)

PRIORITIZE for ERP search:
- News articles about technology implementations
- Company websites mentioning software systems
- Job postings mentioning specific ERP systems
- Case studies and success stories
- Technology vendor websites with customer lists
- Industry publications about tech adoption
- Company press releases about system implementations
- IT consulting firm websites with client mentions
- Business directories with technology details

AVOID for ERP search:
- Pure financial/revenue focused content
- General company descriptions without tech details
- Job postings (unless they mention specific ERP systems)
- Social media posts
- Unrelated technology news

Look specifically for mentions of (if possible): SAP, Oracle, Microsoft Dynamics, Salesforce, NetSuite, Tally, QuickBooks, ERP systems, business software, enterprise systems.

Links to evaluate:
{links_text}

Focus specifically on ERP systems, business software, and technology implementations."""

    try:
        curated = await llm_response_structured_object(prompt, CuratedLinks)
        scored_links = sorted(curated.scored_links, key=lambda x: x.score, reverse=True)
        
        print(f"\t📊 {search_type} curation results:")
        for link in scored_links[:3]:
            print(f"\t   Score {link.score}/10: {link.url}")
            print(f"\t   Reason: {link.reason}")
        
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

async def llm_response_structured_object(prompt: str, schema: BaseModel) -> BaseModel:
    """Gets a structured JSON response from an LLM based on a schema."""
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

async def extract_erp_revenue_info(content: str, company_name: str) -> ERPRevenueInfo:
    """Extract ERP and revenue information from scraped content using LLM."""
    prompt = f"""You are a business analyst extracting ERP system and revenue information from web content about '{company_name}'.

Analyze the following content and extract:

1. ERP SYSTEM: Look for mentions of:
   - Enterprise Resource Planning (ERP) systems
   - Business management software
   - Specific ERP brands: SAP, Oracle, Microsoft Dynamics, Salesforce, NetSuite, Tally, QuickBooks, etc.
   - Implementation partners or consultants
   - You can also infer the ERP systems from social posts, job postings, and other sources.(but they will have low confidence)
   - System integrations

2. REVENUE INFORMATION: Look for:
   - Annual revenue/turnover figures
   - Sales figures
   - Financial performance data
   - Revenue growth percentages
   - Market capitalization

3. COMPANY SIZE INDICATORS: Look for:
   - Company classification (SME, Large Enterprise, etc.)
   - Business scale indicators

CONTENT TO ANALYZE:
{content}

Instructions:
- Be conservative with confidence scores (only give 8-10 for explicit mentions)
- For revenue, include currency and year if available
- Categorize revenue into ranges: Under $1M, $1M-$10M, $10M-$50M, $50M-$100M, $100M+
- If no clear information is found, set fields to null and confidence to 1-3
- Provide reasoning in notes field"""

    try:
        result = await llm_response_structured_object(prompt, ERPRevenueInfo)
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

# --- Main ERP Revenue Research Function ---

async def research_company_erp_revenue(company_name: str, location: str = "", industry: str = "") -> Dict:
    """Research a single company for ERP and revenue information."""
    print(f"\n🔍 Researching: {company_name}")
    
    # Create search context
    search_context = company_name
    if location:
        search_context += f" {location}"
    if industry:
        search_context += f" {industry}"
    
    # Simple two-query search strategy with specific search types
    search_queries = [
        (f"{company_name} revenue", "revenue"),
        (f"{company_name} \"ERP\"", "ERP")
    ]
    
    # Process each query separately and curate
    all_curated_links = []
    
    for query, search_type in search_queries:
        print(f"\t🔎 Processing {search_type} query: {query}")
        results = await web_search(query)
        
        if results:
            # Curate links for this specific search type
            curated_for_query = await curate_links(results, company_name, search_type)
            all_curated_links.extend(curated_for_query)
            print(f"\t✅ Curated {len(curated_for_query)} links for {search_type} search")
    
    # Remove duplicates from final curated list
    curated_links = deduplicate_links(all_curated_links)
    print(f"\t🔗 Total curated links after deduplication: {len(curated_links)}")
    
    # Scrape content from curated links
    all_content = []
    for link in curated_links:
        url = link.get('url')
        if url:
            content = await read_webpage_text(url)
            if content:
                all_content.append(content)
    
    # Combine all content and extract information
    combined_content = "\n\n---\n\n".join(all_content)
    
    if combined_content.strip():
        erp_revenue_info = await extract_erp_revenue_info(combined_content, company_name)
    else:
        print(f"\t⚠️ No content scraped for {company_name}")
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

# --- Main Application ---

async def process_companies_from_file(file_path: str, output_file: str = None):
    """Process companies from CSV/Excel file and generate ERP/revenue report."""
    
    # Read input file
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("File must be CSV or Excel format")
        
        print(f"📊 Loaded {len(df)} companies from {file_path}")
        print(f"📋 Columns found: {list(df.columns)}")
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return
    
    # Validate required columns
    required_col = 'company_name'
    if required_col not in df.columns:
        print(f"❌ Required column '{required_col}' not found in file")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Optional columns
    location_col = 'location' if 'location' in df.columns else None
    industry_col = 'industry' if 'industry' in df.columns else None
    
    # Process each company
    results = []
    total_companies = len(df)
    
    print(f"\n🚀 Starting ERP/Revenue research for {total_companies} companies...\n")
    
    for idx, row in df.iterrows():
        company_name = str(row[required_col]).strip()
        location = str(row[location_col]).strip() if location_col and pd.notna(row[location_col]) else ""
        industry = str(row[industry_col]).strip() if industry_col and pd.notna(row[industry_col]) else ""
        
        print(f"📈 Progress: {idx + 1}/{total_companies}")
        
        try:
            result = await research_company_erp_revenue(company_name, location, industry)
            results.append(result)
            
            # Print quick summary
            erp = result.get('erp_system', 'Not found')
            revenue = result.get('annual_revenue', 'Not found')
            print(f"\t💼 ERP: {erp}")
            print(f"\t💰 Revenue: {revenue}")
            
        except Exception as e:
            print(f"\t❌ Error processing {company_name}: {e}")
            # Add error entry to results
            results.append({
                "company_name": company_name,
                "location": location,
                "industry": industry,
                "erp_system": None,
                "erp_confidence": 0,
                "annual_revenue": None,
                "revenue_confidence": 0,
                "revenue_range": None,
                "company_size": None,
                "business_type": None,
                "notes": f"Error during processing: {str(e)}",
                "sources_found": 0
            })
        
        # Add small delay to be respectful to APIs
        await asyncio.sleep(1)
    
    # Create output DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    if not output_file:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"erp_revenue_results_{timestamp}.xlsx"
    
    try:
        if output_file.endswith('.csv'):
            results_df.to_csv(output_file, index=False)
        else:
            results_df.to_excel(output_file, index=False)
        
        print(f"\n✅ Results saved to: {output_file}")
        
        # Print summary statistics
        print(f"\n📊 SUMMARY:")
        print(f"Total companies processed: {len(results)}")
        print(f"Companies with ERP info found: {len([r for r in results if r['erp_system']])}")
        print(f"Companies with revenue info found: {len([r for r in results if r['annual_revenue']])}")
        
        # Show ERP systems found
        erp_systems = [r['erp_system'] for r in results if r['erp_system']]
        if erp_systems:
            erp_counts = pd.Series(erp_systems).value_counts()
            print(f"\n🖥️ ERP Systems Found:")
            for erp, count in erp_counts.head().items():
                print(f"   {erp}: {count} companies")
        
        # Show revenue ranges
        revenue_ranges = [r['revenue_range'] for r in results if r['revenue_range']]
        if revenue_ranges:
            range_counts = pd.Series(revenue_ranges).value_counts()
            print(f"\n💰 Revenue Ranges:")
            for range_val, count in range_counts.items():
                print(f"   {range_val}: {count} companies")
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")

def main():
    """Main function to run the ERP Revenue Agent."""
    print("🏢 ERP & Revenue Research Agent")
    print("=" * 50)
    
    # Get input file
    while True:
        file_path = input("\n📁 Enter path to CSV/Excel file with company names: ").strip()
        if file_path and Path(file_path).exists():
            break
        print("❌ File not found. Please check the path and try again.")
    
    # Get output file (optional)
    output_file = input("📄 Enter output file name (optional, press Enter for auto-generated): ").strip()
    if not output_file:
        output_file = None
    
    # Confirmation 
    confirm = input(f"\n🚀 Start processing companies from '{file_path}'? (y/n): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("❌ Process cancelled.")
        return
    
    # Run the processing
    try:
        asyncio.run(process_companies_from_file(file_path, output_file))
    except KeyboardInterrupt:
        print("\n\n❌ Process interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ An error occurred: {e}")

if __name__ == "__main__":
    main() 