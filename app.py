"""
SEO Internal Linking Tool
AI-powered semantic analysis for intelligent internal link placement
Supports both E-commerce PLP mode and Blog mode
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import json
import re
import requests
from urllib.parse import urlparse, urljoin
from typing import List, Dict, Tuple, Optional
import networkx as nx
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import xml.etree.ElementTree as ET

# Page config
st.set_page_config(
    page_title="SEO Internal Linking Tool",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_api_key() -> str:
    """Get API key from secrets or session state"""
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    return st.session_state.get('manual_api_key', '')


def parse_url_path(url) -> Dict:
    """Extract category structure from URL"""
    try:
        if url is None or (isinstance(url, float) and np.isnan(url)):
            return {'category': '', 'subcategory': '', 'color': '', 'depth': 0}

        url = str(url)
        from urllib.parse import urlparse
        parsed = urlparse(url)
        path_parts = [p for p in parsed.path.split('/') if p]
        return {
            'category': path_parts[0] if len(path_parts) > 0 else '',
            'subcategory': path_parts[1] if len(path_parts) > 1 else '',
            'color': path_parts[2] if len(path_parts) > 2 else path_parts[1] if len(path_parts) > 1 else '',
            'depth': len(path_parts)
        }
    except:
        return {'category': '', 'subcategory': '', 'color': '', 'depth': 0}


def extract_text_from_html(html: str) -> str:
    """Remove HTML tags and extract plain text"""
    if not html:
        return ""
    text = re.sub(r'<[^>]+>', ' ', html)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ============================================================================
# PLP WHITELIST - Only these URLs are valid link targets
# ============================================================================
PLP_WHITELIST = {
    "https://uk.triumph.com/bras",
    "https://uk.triumph.com/bras/minimizer",
    "https://uk.triumph.com/bras/fuller-cups",
    "https://uk.triumph.com/bras/backless-low-back",
    "https://uk.triumph.com/bras/strapless-multiway",
    "https://uk.triumph.com/bras/push-up-bras",
    "https://uk.triumph.com/bras/t-shirt-bras",
    "https://uk.triumph.com/bras/non-wired-bras",
    "https://uk.triumph.com/bras/bralettes",
    "https://uk.triumph.com/bras/sports-bras",
    "https://uk.triumph.com/bras/lace-bras",
    "https://uk.triumph.com/bras/nursing-bras",
    "https://uk.triumph.com/knickers-panties-briefs",
    "https://uk.triumph.com/knickers-panties-briefs/brazilian",
    "https://uk.triumph.com/knickers-panties-briefs/midi-briefs",
    "https://uk.triumph.com/knickers-panties-briefs/hipster",
    "https://uk.triumph.com/knickers-panties-briefs/maxi-briefs",
    "https://uk.triumph.com/knickers-panties-briefs/thongs",
    "https://uk.triumph.com/knickers-panties-briefs/shorts",
    "https://uk.triumph.com/knickers-panties-briefs/strings",
    "https://uk.triumph.com/shapewear",
    "https://uk.triumph.com/shapewear/bodysuits-all-in-ones",
    "https://uk.triumph.com/shapewear/shaping-briefs",
    "https://uk.triumph.com/shapewear/waist-cinchers-corsets",
    "https://uk.triumph.com/shapewear/open-bust-shapers",
    "https://uk.triumph.com/shapewear/shaping-dresses-skirts",
    "https://uk.triumph.com/shapewear/bras",
    "https://uk.triumph.com/nightwear-loungewear",
    "https://uk.triumph.com/nightwear-loungewear/camisoles",
    "https://uk.triumph.com/nightwear-loungewear/chemises-nighties",
    "https://uk.triumph.com/nightwear-loungewear/pyjamas",
    "https://uk.triumph.com/nightwear-loungewear/bottoms",
    "https://uk.triumph.com/nightwear-loungewear/robes",
    "https://uk.triumph.com/nightwear-loungewear/tops",
    "https://uk.triumph.com/swimwear",
    "https://uk.triumph.com/swimwear/bikinis-tankinis",
    "https://uk.triumph.com/swimwear/bikinis-tankinis/bikini-tops",
    "https://uk.triumph.com/swimwear/bikinis-tankinis/bikini-briefs",
    "https://uk.triumph.com/swimwear/bikinis-tankinis/tankinis",
    "https://uk.triumph.com/swimwear/swimsuits",
    "https://uk.triumph.com/swimwear/beachwear",
    "https://uk.triumph.com/collections",
    "https://uk.triumph.com/collections/amourette",
    "https://uk.triumph.com/collections/body-make-up",
    "https://uk.triumph.com/collections/fit-smart",
    "https://uk.triumph.com/collections/flex-smart",
    "https://uk.triumph.com/collections/signature-sheer",
    "https://uk.triumph.com/collections/wild-peony",
    "https://uk.triumph.com/collections/essential-minimizer",
    "https://uk.triumph.com/collections/ladyform",
    "https://uk.triumph.com/collections/true-shape-sensation",
    "https://uk.triumph.com/colours/beige",
    "https://uk.triumph.com/colours/black",
    "https://uk.triumph.com/colours/blue",
    "https://uk.triumph.com/colours/white",
    "https://uk.triumph.com/colours/pink",
    "https://uk.triumph.com/bras/beige",
    "https://uk.triumph.com/bras/black",
    "https://uk.triumph.com/bras/blue",
    "https://uk.triumph.com/bras/white",
    "https://uk.triumph.com/bras/pink",
    "https://uk.triumph.com/bras/red",
    "https://uk.triumph.com/bras/green",
    "https://uk.triumph.com/bras/purple",
    "https://uk.triumph.com/bras/grey",
    "https://uk.triumph.com/bras/brown",
    "https://uk.triumph.com/bras/orange",
    "https://uk.triumph.com/bras/yellow",
    "https://uk.triumph.com/knickers-panties-briefs/beige",
    "https://uk.triumph.com/knickers-panties-briefs/black",
    "https://uk.triumph.com/knickers-panties-briefs/blue",
    "https://uk.triumph.com/knickers-panties-briefs/white",
    "https://uk.triumph.com/knickers-panties-briefs/pink",
    "https://uk.triumph.com/knickers-panties-briefs/red",
    "https://uk.triumph.com/knickers-panties-briefs/green",
    "https://uk.triumph.com/knickers-panties-briefs/purple",
    "https://uk.triumph.com/knickers-panties-briefs/grey",
    "https://uk.triumph.com/shapewear/beige",
    "https://uk.triumph.com/shapewear/black",
    "https://uk.triumph.com/shapewear/white",
    "https://uk.triumph.com/nightwear-loungewear/beige",
    "https://uk.triumph.com/nightwear-loungewear/black",
    "https://uk.triumph.com/nightwear-loungewear/blue",
    "https://uk.triumph.com/nightwear-loungewear/white",
    "https://uk.triumph.com/nightwear-loungewear/pink",
    "https://uk.triumph.com/nightwear-loungewear/red",
    "https://uk.triumph.com/nightwear-loungewear/grey",
    "https://uk.triumph.com/swimwear/beige",
    "https://uk.triumph.com/swimwear/black",
    "https://uk.triumph.com/swimwear/blue",
    "https://uk.triumph.com/swimwear/white",
    "https://uk.triumph.com/swimwear/pink",
    "https://uk.triumph.com/swimwear/green",
    "https://uk.triumph.com/swimwear/orange",
}


def parse_sitemap(file_content: bytes, filename: str) -> List[str]:
    """Parse sitemap from JSON, TXT, or XML format"""
    urls = []
    content = file_content.decode('utf-8')

    try:
        # First, try to detect format by content, not just extension
        content_stripped = content.strip()

        if content_stripped.startswith('[') or content_stripped.startswith('{'):
            # Looks like JSON
            data = json.loads(content)
            if isinstance(data, list):
                urls = [u for u in data if isinstance(u, str) and u.startswith('http')]
            elif isinstance(data, dict) and 'urls' in data:
                urls = [u for u in data['urls'] if isinstance(u, str) and u.startswith('http')]

        elif content_stripped.startswith('<?xml') or '<urlset' in content_stripped or '<loc>' in content_stripped:
            # XML sitemap
            loc_pattern = re.compile(r'<loc>(.*?)</loc>', re.IGNORECASE)
            urls = loc_pattern.findall(content)

        else:
            # TXT - one URL per line (default fallback)
            urls = [line.strip() for line in content.split('\n')
                    if line.strip().startswith('http')]

    except Exception as e:
        # If JSON parsing fails, try as plain text
        try:
            urls = [line.strip() for line in content.split('\n')
                    if line.strip().startswith('http')]
        except:
            st.warning(f"Error parsing sitemap: {str(e)}")

    return urls


def extract_keyword_from_url(url: str) -> str:
    """Extract a keyword/description from URL path"""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        path_parts = [p for p in parsed.path.split('/') if p]

        if not path_parts:
            return ""

        # Take last 2-3 meaningful parts and convert to readable text
        meaningful = [p.replace('-', ' ').replace('_', ' ') for p in path_parts[-3:]]
        return ' '.join(meaningful)
    except:
        return ""


def is_plp_url(url: str) -> bool:
    """
    Check if URL is a Product Listing Page (PLP) using the official whitelist.
    Only URLs in PLP_WHITELIST are valid link targets.
    """
    if not url:
        return False

    # Normalize URL (remove trailing slash, lowercase for comparison)
    url_normalized = url.rstrip('/').lower()

    # Check against whitelist (also normalized)
    for plp_url in PLP_WHITELIST:
        if plp_url.rstrip('/').lower() == url_normalized:
            return True

    return False


# ============================================================================
# BLOG MODE FUNCTIONS
# ============================================================================

def fetch_sitemap_urls(sitemap_url: str, prefix_filter: str = "", max_urls: int = 300) -> List[Dict]:
    """
    Fetch URLs from an XML sitemap.
    Returns list of dicts with url, title (from lastmod or path), and description.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; SEOInternalLinkingBot/1.0)'
        }
        response = requests.get(sitemap_url, headers=headers, timeout=30)
        response.raise_for_status()

        urls = []
        content = response.text

        # Parse XML sitemap
        # Handle namespace
        content_clean = re.sub(r'\sxmlns="[^"]+"', '', content, count=1)

        try:
            root = ET.fromstring(content_clean)
        except ET.ParseError:
            # Fallback: extract URLs with regex
            loc_matches = re.findall(r'<loc>(.*?)</loc>', content)
            for url in loc_matches[:max_urls]:
                if prefix_filter and prefix_filter not in url:
                    continue
                urls.append({
                    'url': url.strip(),
                    'title': extract_title_from_url(url),
                    'description': ''
                })
            return urls

        # Find all url elements
        for url_elem in root.findall('.//url'):
            loc = url_elem.find('loc')
            if loc is not None and loc.text:
                url = loc.text.strip()

                # Apply prefix filter
                if prefix_filter and prefix_filter not in url:
                    continue

                # Try to get lastmod for context
                lastmod = url_elem.find('lastmod')
                lastmod_text = lastmod.text if lastmod is not None else ''

                urls.append({
                    'url': url,
                    'title': extract_title_from_url(url),
                    'description': f'Last modified: {lastmod_text}' if lastmod_text else ''
                })

                if len(urls) >= max_urls:
                    break

        return urls

    except Exception as e:
        st.error(f"Error fetching sitemap: {str(e)}")
        return []


def extract_title_from_url(url: str) -> str:
    """Extract a readable title from URL path"""
    try:
        parsed = urlparse(url)
        path_parts = [p for p in parsed.path.split('/') if p]

        if not path_parts:
            return parsed.netloc

        # Take the last meaningful part and convert to title
        last_part = path_parts[-1]
        # Remove file extension
        last_part = re.sub(r'\.(html?|php|aspx?)$', '', last_part, flags=re.IGNORECASE)
        # Convert dashes/underscores to spaces and title case
        title = last_part.replace('-', ' ').replace('_', ' ').title()
        return title
    except:
        return url


def scrape_webpage_content(url: str) -> Dict:
    """
    Scrape the main content from a webpage.
    Returns dict with title, description, and body text.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        html = response.text

        # Extract title
        title_match = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else ''

        # Extract meta description
        desc_match = re.search(r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']*)["\']', html, re.IGNORECASE)
        if not desc_match:
            desc_match = re.search(r'<meta[^>]*content=["\']([^"\']*)["\'][^>]*name=["\']description["\']', html, re.IGNORECASE)
        description = desc_match.group(1).strip() if desc_match else ''

        # Extract body content - remove script, style, nav, header, footer
        body_match = re.search(r'<body[^>]*>(.*?)</body>', html, re.IGNORECASE | re.DOTALL)
        if body_match:
            body = body_match.group(1)
        else:
            body = html

        # Remove unwanted elements
        body = re.sub(r'<script[^>]*>.*?</script>', '', body, flags=re.IGNORECASE | re.DOTALL)
        body = re.sub(r'<style[^>]*>.*?</style>', '', body, flags=re.IGNORECASE | re.DOTALL)
        body = re.sub(r'<nav[^>]*>.*?</nav>', '', body, flags=re.IGNORECASE | re.DOTALL)
        body = re.sub(r'<header[^>]*>.*?</header>', '', body, flags=re.IGNORECASE | re.DOTALL)
        body = re.sub(r'<footer[^>]*>.*?</footer>', '', body, flags=re.IGNORECASE | re.DOTALL)
        body = re.sub(r'<aside[^>]*>.*?</aside>', '', body, flags=re.IGNORECASE | re.DOTALL)
        body = re.sub(r'<!--.*?-->', '', body, flags=re.DOTALL)

        # Keep the HTML structure for link insertion, but also get plain text
        plain_text = re.sub(r'<[^>]+>', ' ', body)
        plain_text = re.sub(r'\s+', ' ', plain_text).strip()

        return {
            'title': title,
            'description': description,
            'html_content': body.strip(),
            'plain_text': plain_text[:10000],  # Limit for API
            'url': url
        }

    except Exception as e:
        st.error(f"Error scraping {url}: {str(e)}")
        return {
            'title': '',
            'description': '',
            'html_content': '',
            'plain_text': '',
            'url': url
        }


def get_blog_link_suggestions(
    target_content: Dict,
    candidate_pages: List[Dict],
    api_key: str,
    max_links: int = 5,
    model: str = "gpt-4o"
) -> List[Dict]:
    """
    Use GPT to suggest internal links for a blog post.
    Similar to PLP mode but optimized for blog content.
    """
    client = OpenAI(api_key=api_key)

    # Prepare candidates summary
    candidates_text = "\n".join([
        f"- URL: {p['url']}\n  Title: {p['title']}\n  Description: {p.get('description', '')}"
        for p in candidate_pages[:30]
    ])

    prompt = f"""You are an expert SEO content strategist analyzing a blog post for internal linking opportunities.

TARGET BLOG POST:
URL: {target_content['url']}
Title: {target_content['title']}

CONTENT:
{target_content['plain_text'][:4000]}

AVAILABLE PAGES TO LINK TO:
{candidates_text}

YOUR TASK:
Identify {max_links} optimal places to insert internal links that will:
1. Improve user experience by connecting related content
2. Boost topical authority and SEO
3. Feel natural and helpful, not forced

RULES:
1. ONLY use URLs from the AVAILABLE PAGES list - never invent URLs
2. NEVER link to the target post itself
3. Find EXACT phrases in the content that make good anchor texts
4. Prefer descriptive anchor texts (2-5 words) over single generic words
5. Don't link words like "here", "click", "this", "read more"
6. Each URL should only be used ONCE
7. Distribute links throughout the article, not clustered
8. If no good match exists for a URL, skip it - quality over quantity

Return your suggestions as a JSON array:
[
  {{
    "anchor_text": "exact phrase from the article to make into a link",
    "target_url": "URL from the available list",
    "reason": "brief explanation of why this link adds value"
  }}
]

Return ONLY the JSON array, no other text."""

    try:
        if model.startswith("gpt-5"):
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_completion_tokens=1500
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500
            )

        result = response.choices[0].message.content.strip()

        # Parse JSON from response
        if result.startswith("```"):
            result = re.sub(r'^```json?\n?', '', result)
            result = re.sub(r'\n?```$', '', result)

        suggestions = json.loads(result)
        return suggestions

    except Exception as e:
        st.warning(f"AI suggestion error: {str(e)}")
        return []


def insert_blog_links(html_content: str, suggestions: List[Dict], source_url: str = "") -> Tuple[str, List[Dict]]:
    """Insert links into blog HTML content based on AI suggestions"""
    if not html_content or not suggestions:
        return html_content, []

    result = html_content
    links_inserted = []
    used_anchors = set()
    used_urls = set()

    for suggestion in suggestions:
        anchor = suggestion.get('anchor_text', '')
        target_url = suggestion.get('target_url', '')
        reason = suggestion.get('reason', '')

        if not anchor or not target_url:
            continue

        # No self-linking
        if target_url == source_url:
            continue

        # No duplicate anchors or URLs
        if anchor.lower() in used_anchors or target_url in used_urls:
            continue

        # Split by existing links to avoid nesting
        parts = re.split(r'(<a\s[^>]*>.*?</a>)', result, flags=re.IGNORECASE | re.DOTALL)

        found = False
        new_parts = []

        for part in parts:
            if found or part.lower().startswith('<a ') or part.lower().startswith('<a>'):
                new_parts.append(part)
            else:
                pattern = re.compile(re.escape(anchor), re.IGNORECASE)
                match = pattern.search(part)

                if match:
                    original_text = match.group(0)
                    replacement = f'<a href="{target_url}">{original_text}</a>'
                    new_part = part[:match.start()] + replacement + part[match.end():]
                    new_parts.append(new_part)
                    found = True

                    links_inserted.append({
                        'anchor_text': original_text,
                        'target_url': target_url,
                        'reason': reason
                    })
                    used_anchors.add(anchor.lower())
                    used_urls.add(target_url)
                else:
                    new_parts.append(part)

        if found:
            result = ''.join(new_parts)

    return result, links_inserted


def urls_are_semantically_coherent(source_url: str, target_url: str) -> bool:
    """
    Check if linking from source to target makes semantic sense.
    Avoid linking beige to black, bras to completely unrelated categories, etc.
    """
    source_parsed = parse_url_path(source_url)
    target_parsed = parse_url_path(target_url)

    source_path = source_url.lower()
    target_path = target_url.lower()

    # Extract color from URLs if present
    colors = ['beige', 'black', 'white', 'brown', 'blue', 'red', 'pink', 'grey', 'gray', 'green', 'nude', 'cream', 'tan']

    source_color = None
    target_color = None

    for color in colors:
        if f'/{color}' in source_path or source_path.endswith(f'/{color}'):
            source_color = color
        if f'/{color}' in target_path or target_path.endswith(f'/{color}'):
            target_color = color

    # If both have colors and they're completely different (not similar tones), flag as incoherent
    # Allow: beige->nude, beige->cream, beige->tan, beige->brown (similar neutral tones)
    # Disallow: beige->black, beige->blue, beige->red
    neutral_tones = {'beige', 'nude', 'cream', 'tan', 'brown'}

    if source_color and target_color:
        if source_color in neutral_tones and target_color not in neutral_tones:
            return False
        if source_color not in neutral_tones and target_color in neutral_tones:
            # Allow linking from colored to neutral
            pass
        elif source_color != target_color:
            # Different non-neutral colors
            source_neutral = source_color in neutral_tones
            target_neutral = target_color in neutral_tones
            if not (source_neutral and target_neutral):
                return False

    return True


# ============================================================================
# AI-POWERED LINK SUGGESTION
# ============================================================================

def get_ai_link_suggestions(
    source_text: str,
    source_url: str,
    target_pages: List[Dict],
    api_key: str,
    max_links: int = 5,
    model: str = "gpt-4o"
) -> List[Dict]:
    """
    Use GPT to analyze the source text and suggest intelligent link placements.
    This mimics how an expert SEO would identify linking opportunities.
    """
    client = OpenAI(api_key=api_key)

    # Prepare target pages summary
    targets_summary = "\n".join([
        f"- URL: {p['url']}\n  Keyword: {p['keyword']}\n  Category: {p['category']}"
        for p in target_pages[:30]  # Limit to top 30 most relevant
    ])

    prompt = f"""You are an expert SEO specialist analyzing content for internal linking opportunities.

SOURCE PAGE: {source_url}

SOURCE CONTENT (Bottom SEO Text):
{source_text[:3000]}

AVAILABLE TARGET PAGES TO LINK TO (Category Pages Only):
{targets_summary}

YOUR TASK:
Analyze the source content and identify {max_links} optimal places to insert internal links.

STRICT RULES:
1. ONLY use URLs from the AVAILABLE TARGET PAGES list above - never invent URLs
2. NEVER link to the source page itself (no self-linking)
3. ONLY link to category pages (PLPs) - never to product pages ending in .html or with numeric codes
4. Maintain color coherence: if source is about "beige", only link to beige/nude/cream/tan/brown pages, never to black/blue/red etc.
5. Each target URL should only be used ONCE
6. The anchor text MUST exist EXACTLY as written in the source content

QUALITY GUIDELINES:
- Find natural phrases IN THE EXISTING TEXT that make good anchor texts
- Prefer descriptive anchor texts (2-4 words) over single words
- Don't link generic words like "here", "click", "this", "our", "the"
- Ensure links add value for the reader
- Distribute links throughout the content

Return your suggestions as a JSON array:
[
  {{
    "anchor_text": "exact text from source to make into link",
    "target_url": "URL from the available list above",
    "reason": "brief SEO explanation of why this link is valuable"
  }}
]

Return ONLY the JSON array, no other text."""

    try:
        # GPT-5.x uses max_completion_tokens, older models use max_tokens
        if model.startswith("gpt-5"):
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_completion_tokens=1500
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500
            )

        result = response.choices[0].message.content.strip()

        # Parse JSON from response
        if result.startswith("```"):
            result = re.sub(r'^```json?\n?', '', result)
            result = re.sub(r'\n?```$', '', result)

        suggestions = json.loads(result)
        return suggestions

    except Exception as e:
        st.warning(f"AI suggestion error for {source_url}: {str(e)}")
        return []


def insert_ai_suggested_links(html_text: str, suggestions: List[Dict], source_url: str = "") -> Tuple[str, List[Dict]]:
    """Insert links based on AI suggestions with validation"""
    if not html_text or not suggestions:
        return html_text, []

    result = html_text
    links_inserted = []
    used_anchors = set()
    used_urls = set()

    for suggestion in suggestions:
        anchor = suggestion.get('anchor_text', '')
        target_url = suggestion.get('target_url', '')
        reason = suggestion.get('reason', '')

        if not anchor or not target_url:
            continue

        # VALIDATION 1: No self-linking
        if target_url == source_url:
            continue

        # VALIDATION 2: Only PLP URLs
        if not is_plp_url(target_url):
            continue

        # VALIDATION 3: Semantic coherence
        if source_url and not urls_are_semantically_coherent(source_url, target_url):
            continue

        # VALIDATION 4: No duplicate anchors
        if anchor.lower() in used_anchors:
            continue

        # VALIDATION 5: No duplicate target URLs
        if target_url in used_urls:
            continue

        # Find anchor in text, avoiding existing links
        # Strategy: split by <a>...</a> tags, only search in non-link parts
        parts = re.split(r'(<a\s[^>]*>.*?</a>)', result, flags=re.IGNORECASE | re.DOTALL)

        found = False
        new_parts = []

        for part in parts:
            if found or part.startswith('<a ') or part.startswith('<a>'):
                # Already found or this is an existing link - keep as is
                new_parts.append(part)
            else:
                # Search for anchor in this non-link part
                pattern = re.compile(re.escape(anchor), re.IGNORECASE)
                match = pattern.search(part)

                if match:
                    # Replace first occurrence
                    original_text = match.group(0)
                    replacement = f'<a href="{target_url}">{original_text}</a>'
                    new_part = part[:match.start()] + replacement + part[match.end():]
                    new_parts.append(new_part)
                    found = True

                    links_inserted.append({
                        'anchor_text': original_text,
                        'target_url': target_url,
                        'reason': reason
                    })
                    used_anchors.add(anchor.lower())
                    used_urls.add(target_url)
                else:
                    new_parts.append(part)

        if found:
            result = ''.join(new_parts)

    return result, links_inserted


# ============================================================================
# EMBEDDING FUNCTIONS
# ============================================================================

@st.cache_data(show_spinner=False)
def get_embeddings_batch(_texts: List[str], api_key: str, model: str = "text-embedding-3-small") -> np.ndarray:
    """Get embeddings for a batch of texts using OpenAI API"""
    client = OpenAI(api_key=api_key)

    # Filter empty texts
    texts = list(_texts)  # Convert to list to avoid issues
    valid_indices = [i for i, t in enumerate(texts) if t and len(str(t).strip()) > 0]
    valid_texts = [str(texts[i])[:8000] for i in valid_indices]  # Limit text length

    if not valid_texts:
        return np.zeros((len(texts), 1536))

    # Batch requests
    batch_size = 100
    all_embeddings = []

    for i in range(0, len(valid_texts), batch_size):
        batch = valid_texts[i:i+batch_size]
        response = client.embeddings.create(
            model=model,
            input=batch
        )
        batch_embeddings = [e.embedding for e in response.data]
        all_embeddings.extend(batch_embeddings)

    # Map back to original indices
    result = np.zeros((len(texts), len(all_embeddings[0]) if all_embeddings else 1536))
    for idx, emb in zip(valid_indices, all_embeddings):
        result[idx] = emb

    return result


def calculate_semantic_similarity(embeddings: np.ndarray) -> np.ndarray:
    """Calculate pairwise cosine similarity between all embeddings"""
    return cosine_similarity(embeddings)


# ============================================================================
# PAGERANK FUNCTIONS
# ============================================================================

def build_link_graph(df: pd.DataFrame, similarity_matrix: np.ndarray,
                     similarity_threshold: float = 0.5) -> nx.DiGraph:
    """Build a directed graph based on semantic similarity"""
    G = nx.DiGraph()

    for idx, row in df.iterrows():
        url = row.get('URL', '')
        if pd.isna(url):
            url = ''
        G.add_node(idx,
                   url=str(url),
                   title=str(row.get('Title', '')),
                   category=parse_url_path(url)['category'])

    n = len(df)
    for i in range(n):
        for j in range(n):
            if i != j and similarity_matrix[i, j] >= similarity_threshold:
                G.add_edge(i, j, weight=float(similarity_matrix[i, j]))

    return G


def calculate_pagerank(G: nx.DiGraph, damping: float = 0.85) -> Dict[int, float]:
    """Calculate PageRank for all nodes"""
    if len(G.nodes()) == 0:
        return {}
    try:
        return nx.pagerank(G, alpha=damping)
    except:
        return {n: 1.0/len(G.nodes()) for n in G.nodes()}


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_network_graph(
    G: nx.DiGraph,
    df: pd.DataFrame,
    pagerank_scores: Dict[int, float]
) -> go.Figure:
    """Create an interactive network visualization"""

    if len(G.nodes()) == 0:
        return go.Figure()

    pos = nx.spring_layout(G, k=2, iterations=50)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_text = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        pr = pagerank_scores.get(node, 0)
        node_colors.append(pr)
        node_sizes.append(10 + pr * 500)

        url = str(df.iloc[node].get('URL', ''))
        category = parse_url_path(url)['category']
        node_text.append(f"URL: {url}<br>Category: {category}<br>PageRank: {pr:.4f}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            color=node_colors,
            size=node_sizes,
            colorbar=dict(
                thickness=15,
                title=dict(text='PageRank', side='right'),
                xanchor='left'
            ),
            line_width=2
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title='Internal Link Network',
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20, l=5, r=5, t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                   ))

    return fig


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown('<p class="main-header">🔗 SEO Internal Linking Tool</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered semantic analysis for intelligent internal link placement</p>', unsafe_allow_html=True)

    # Initialize session state
    if 'manual_api_key' not in st.session_state:
        st.session_state['manual_api_key'] = ''
    if 'app_mode' not in st.session_state:
        st.session_state['app_mode'] = 'E-commerce (Triumph PLP)'

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Mode selection
        app_mode = st.selectbox(
            "🎯 Mode",
            [
                "E-commerce (Triumph PLP)",
                "Blog Posts"
            ],
            index=0 if st.session_state['app_mode'] == 'E-commerce (Triumph PLP)' else 1,
            help="Choose the type of content you want to optimize"
        )
        st.session_state['app_mode'] = app_mode

        st.divider()

        # API Key handling
        has_secret_key = "OPENAI_API_KEY" in st.secrets

        if has_secret_key:
            st.success("✓ API key loaded from secrets")
            api_key = st.secrets["OPENAI_API_KEY"]
        else:
            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Your OpenAI API key for AI-powered analysis",
                key="api_key_input"
            )
            if api_key:
                st.session_state['manual_api_key'] = api_key

        st.divider()

        # Parameters
        st.subheader("Parameters")

        max_links = st.slider(
            "Max links per page",
            min_value=1, max_value=10, value=5,
            help="Maximum number of internal links to insert per page"
        )

        # Only show these for E-commerce mode
        if app_mode == "E-commerce (Triumph PLP)":
            min_similarity = st.slider(
                "Min similarity threshold",
                min_value=0.3, max_value=0.9, value=0.5, step=0.05,
                help="Minimum semantic similarity to consider for linking"
            )

            damping_factor = st.slider(
                "PageRank damping factor",
                min_value=0.5, max_value=0.95, value=0.85, step=0.05,
                help="Standard is 0.85 - represents probability of following a link"
            )

            use_ai_suggestions = st.checkbox(
                "🤖 Use AI for smart suggestions",
                value=True,
                help="Use GPT to analyze context and suggest optimal anchor texts like an SEO expert"
            )
        else:
            # Blog mode always uses AI
            min_similarity = 0.5
            damping_factor = 0.85
            use_ai_suggestions = True
            st.info("🤖 Blog mode uses AI-powered semantic analysis")

        ai_model = st.selectbox(
            "AI Model",
            [
                "gpt-4o",
                "gpt-4o-mini",
            ],
            index=0,
            help="GPT-4o is the most intelligent model. GPT-4o-mini for faster/cheaper processing."
        )

    # =========================================================================
    # BLOG MODE
    # =========================================================================
    if app_mode == "Blog Posts":
        tab1, tab2 = st.tabs(["📤 Process Blog", "📥 Results"])

        with tab1:
            st.header("Add Internal Links to Blog Posts")

            st.markdown("""
            This mode helps you automatically add internal links to your blog posts using AI.

            **How it works:**
            1. Enter your sitemap URL and optional path filter
            2. Enter the URL of the blog post you want to optimize
            3. The AI will find relevant pages and suggest natural link placements
            """)

            col1, col2 = st.columns(2)

            with col1:
                sitemap_url = st.text_input(
                    "Sitemap URL",
                    placeholder="https://yourdomain.com/sitemap.xml",
                    help="Your XML sitemap URL containing all blog posts"
                )

                prefix_filter = st.text_input(
                    "Path filter (optional)",
                    placeholder="/blog/",
                    help="Only include URLs containing this path (e.g., /blog/, /articles/)"
                )

                max_sitemap_urls = st.number_input(
                    "Max URLs to fetch",
                    min_value=10,
                    max_value=500,
                    value=300,
                    help="Maximum number of URLs to fetch from sitemap"
                )

            with col2:
                target_url = st.text_input(
                    "Target Post URL",
                    placeholder="https://yourdomain.com/blog/your-post",
                    help="The blog post you want to add internal links to"
                )

                st.markdown("---")
                st.markdown("**Preview Settings**")
                show_original = st.checkbox("Show original content", value=True)

            # Check if we can proceed
            can_process = bool(api_key) and bool(sitemap_url) and bool(target_url)

            if not api_key:
                st.warning("⚠️ Please enter your OpenAI API key in the sidebar")
            elif not sitemap_url:
                st.info("👆 Enter your sitemap URL to get started")
            elif not target_url:
                st.info("👆 Enter the target blog post URL")

            if st.button("🚀 Process Blog Post", type="primary", disabled=not can_process):
                progress_bar = st.progress(0)
                status = st.empty()

                # Step 1: Fetch sitemap
                status.text("📥 Fetching sitemap URLs...")
                sitemap_pages = fetch_sitemap_urls(sitemap_url, prefix_filter, max_sitemap_urls)

                if not sitemap_pages:
                    st.error("❌ Could not fetch any URLs from sitemap")
                    st.stop()

                st.success(f"✅ Found {len(sitemap_pages)} URLs in sitemap")
                progress_bar.progress(20)

                # Step 2: Scrape target page
                status.text("📄 Scraping target blog post...")
                target_content = scrape_webpage_content(target_url)

                if not target_content['plain_text']:
                    st.error("❌ Could not scrape target page content")
                    st.stop()

                st.success(f"✅ Scraped: {target_content['title']}")
                progress_bar.progress(40)

                # Step 3: Get embeddings for semantic search
                status.text("🧠 Finding semantically relevant pages...")

                # Prepare texts for embedding
                target_text = f"{target_content['title']} {target_content['description']} {target_content['plain_text'][:2000]}"
                candidate_texts = [
                    f"{p['title']} {p.get('description', '')}"
                    for p in sitemap_pages
                ]

                # Get embeddings
                all_texts = [target_text] + candidate_texts
                try:
                    embeddings = get_embeddings_batch(all_texts, api_key)
                except Exception as e:
                    st.error(f"Error getting embeddings: {str(e)}")
                    st.stop()

                progress_bar.progress(60)

                # Calculate similarity and get top candidates
                target_embedding = embeddings[0:1]
                candidate_embeddings = embeddings[1:]

                similarities = cosine_similarity(target_embedding, candidate_embeddings)[0]

                # Get top 30 most similar (excluding the target itself)
                top_indices = np.argsort(similarities)[::-1]

                top_candidates = []
                for idx in top_indices[:30]:
                    page = sitemap_pages[idx]
                    if page['url'] != target_url:  # Exclude self
                        page['similarity'] = float(similarities[idx])
                        top_candidates.append(page)

                st.success(f"✅ Found {len(top_candidates)} relevant pages")
                progress_bar.progress(70)

                # Step 4: Get AI link suggestions
                status.text("🤖 AI is analyzing content for link opportunities...")

                suggestions = get_blog_link_suggestions(
                    target_content=target_content,
                    candidate_pages=top_candidates,
                    api_key=api_key,
                    max_links=max_links,
                    model=ai_model
                )

                progress_bar.progress(85)

                # Step 5: Insert links
                status.text("🔗 Inserting links...")

                updated_html, links_inserted = insert_blog_links(
                    target_content['html_content'],
                    suggestions,
                    target_url
                )

                progress_bar.progress(100)
                status.text("✅ Done!")

                # Store results
                st.session_state['blog_results'] = {
                    'target_url': target_url,
                    'target_title': target_content['title'],
                    'original_html': target_content['html_content'],
                    'updated_html': updated_html,
                    'links_inserted': links_inserted,
                    'top_candidates': top_candidates[:10],
                    'show_original': show_original
                }

                st.success(f"🎉 Inserted {len(links_inserted)} internal links!")
                st.balloons()

        with tab2:
            st.header("Results")

            if 'blog_results' not in st.session_state:
                st.info("👆 Process a blog post first to see results")
            else:
                results = st.session_state['blog_results']

                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Links Inserted", len(results['links_inserted']))
                with col2:
                    st.metric("Candidates Found", len(results['top_candidates']))
                with col3:
                    st.metric("Target", results['target_title'][:30] + "...")

                st.divider()

                # Links inserted
                st.subheader("🔗 Links Inserted")
                if results['links_inserted']:
                    for i, link in enumerate(results['links_inserted'], 1):
                        with st.expander(f"{i}. {link['anchor_text']} → {link['target_url'].split('/')[-1]}"):
                            st.markdown(f"**Anchor:** {link['anchor_text']}")
                            st.markdown(f"**URL:** {link['target_url']}")
                            st.markdown(f"**Reason:** {link['reason']}")
                else:
                    st.info("No links were inserted. The AI couldn't find suitable placements.")

                st.divider()

                # Content comparison
                st.subheader("📝 Content Preview")

                if results.get('show_original', True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Original HTML:**")
                        st.code(results['original_html'][:3000] + ('...' if len(results['original_html']) > 3000 else ''), language='html')
                    with col2:
                        st.markdown("**Updated HTML:**")
                        st.code(results['updated_html'][:3000] + ('...' if len(results['updated_html']) > 3000 else ''), language='html')
                else:
                    st.markdown("**Updated HTML:**")
                    st.code(results['updated_html'][:5000] + ('...' if len(results['updated_html']) > 5000 else ''), language='html')

                st.divider()

                # Download
                st.subheader("📥 Download")
                col1, col2 = st.columns(2)

                with col1:
                    st.download_button(
                        label="📥 Download Updated HTML",
                        data=results['updated_html'],
                        file_name=f"updated_{results['target_url'].split('/')[-1]}.html",
                        mime="text/html",
                        type="primary"
                    )

                with col2:
                    # JSON export
                    export_data = {
                        'target_url': results['target_url'],
                        'target_title': results['target_title'],
                        'links_inserted': results['links_inserted'],
                        'updated_html': results['updated_html']
                    }
                    st.download_button(
                        label="📥 Download JSON",
                        data=json.dumps(export_data, indent=2),
                        file_name="blog_links_result.json",
                        mime="application/json"
                    )

        return  # End of Blog Mode

    # =========================================================================
    # E-COMMERCE MODE (Original Triumph PLP mode)
    # =========================================================================
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "📊 Analysis", "📥 Export"])

    with tab1:
        st.header("Upload your data")

        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            st.subheader("SEO Data (XLSX)")
            xlsx_file = st.file_uploader(
                "Upload your SEO keyword analysis file",
                type=['xlsx'],
                help="Excel file with columns: URL, Primary Keyword, Related Keyword, Bottom SEO Text"
            )

        with col2:
            st.subheader("Sitemap (Optional)")
            sitemap_file = st.file_uploader(
                "Upload sitemap for additional targets",
                type=['json', 'txt', 'xml'],
                help="JSON array of URLs, TXT (one URL per line), or XML sitemap. Adds more link targets."
            )

        with col3:
            st.subheader("Info")
            if use_ai_suggestions:
                st.info("🤖 **AI Mode**\nGPT analyzes context for smart links")
            else:
                st.info("📊 **Basic Mode**\nKeyword matching only")

        # Parse sitemap if provided
        sitemap_urls = []
        if sitemap_file:
            sitemap_urls = parse_sitemap(sitemap_file.getvalue(), sitemap_file.name)
            if sitemap_urls:
                st.success(f"✅ Loaded {len(sitemap_urls)} URLs from sitemap")
            else:
                st.warning("⚠️ Could not parse sitemap or no valid URLs found")

        if xlsx_file:
            df = pd.read_excel(xlsx_file)
            st.success(f"✅ Loaded {len(df)} rows from Excel file")

            with st.expander("Preview data"):
                st.dataframe(df.head(10))

            required_cols = ['URL', 'Primary Keyword (Color + Type)', 'Bottom SEO Text']
            missing_cols = [c for c in required_cols if c not in df.columns]

            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                st.stop()

            # Check API key availability
            can_process = bool(api_key)

            if not can_process:
                st.warning("⚠️ Please enter your OpenAI API key in the sidebar to process data")

            if st.button("🚀 Process Data", type="primary", disabled=not can_process):

                st.session_state['df'] = df
                st.session_state['params'] = {
                    'max_links': max_links,
                    'min_similarity': min_similarity,
                    'damping_factor': damping_factor,
                    'use_ai': use_ai_suggestions
                }

                progress_bar = st.progress(0)
                status = st.empty()

                # Step 1: Prepare texts
                status.text("📝 Preparing content for analysis...")
                texts = []
                for idx, row in df.iterrows():
                    text_parts = []
                    if pd.notna(row.get('Title')):
                        text_parts.append(str(row['Title']))
                    if pd.notna(row.get('Primary Keyword (Color + Type)')):
                        text_parts.append(str(row['Primary Keyword (Color + Type)']))
                    if pd.notna(row.get('Bottom SEO Text')):
                        text_parts.append(extract_text_from_html(str(row['Bottom SEO Text']))[:1000])
                    texts.append(' '.join(text_parts))

                progress_bar.progress(10)

                # Step 2: Get embeddings
                status.text("🧠 Generating semantic embeddings...")
                try:
                    embeddings = get_embeddings_batch(texts, api_key)
                    st.session_state['embeddings'] = embeddings
                except Exception as e:
                    st.error(f"Error getting embeddings: {str(e)}")
                    st.stop()

                progress_bar.progress(30)

                # Step 3: Calculate similarity
                status.text("🔄 Calculating semantic similarity...")
                similarity_matrix = calculate_semantic_similarity(embeddings)
                st.session_state['similarity_matrix'] = similarity_matrix

                progress_bar.progress(40)

                # Step 4: Build graph and calculate PageRank
                status.text("📈 Building link graph and calculating PageRank...")
                G = build_link_graph(df, similarity_matrix, min_similarity)
                pagerank_scores = calculate_pagerank(G, damping_factor)
                st.session_state['graph'] = G
                st.session_state['pagerank'] = pagerank_scores

                progress_bar.progress(50)

                # Step 5: Process each page
                status.text("🔗 Analyzing content and generating link suggestions...")
                results = []
                total_links = 0

                # Prepare target pages list from XLSX
                all_targets = []
                xlsx_urls = set()
                for idx, row in df.iterrows():
                    url = row.get('URL', '')
                    if pd.isna(url):
                        continue
                    url_str = str(url)
                    xlsx_urls.add(url_str)
                    all_targets.append({
                        'idx': idx,
                        'url': url_str,
                        'keyword': str(row.get('Primary Keyword (Color + Type)', '')),
                        'category': parse_url_path(url)['category'],
                        'pagerank': pagerank_scores.get(idx, 0),
                        'source': 'xlsx'
                    })

                # Add sitemap URLs as additional targets (if not already in XLSX)
                if sitemap_urls:
                    sitemap_added = 0
                    for url in sitemap_urls:
                        if url not in xlsx_urls:
                            all_targets.append({
                                'idx': -1,  # Not in dataframe
                                'url': url,
                                'keyword': extract_keyword_from_url(url),
                                'category': parse_url_path(url)['category'],
                                'pagerank': 0.001,  # Low default pagerank for sitemap URLs
                                'source': 'sitemap'
                            })
                            sitemap_added += 1
                    if sitemap_added > 0:
                        status.text(f"📍 Added {sitemap_added} additional targets from sitemap")

                for idx in range(len(df)):
                    row = df.iloc[idx].to_dict()
                    source_url = str(row.get('URL', ''))
                    source_text = str(row.get('Bottom SEO Text', ''))

                    if not source_text or pd.isna(row.get('Bottom SEO Text')):
                        results.append({
                            **row,
                            'Bottom SEO Text (with links)': '',
                            'Links Inserted': '',
                            'Links Count': 0,
                            'AI Reasoning': ''
                        })
                        continue

                    # Get top similar pages as candidates from XLSX
                    similarities = similarity_matrix[idx]
                    top_indices = set(np.argsort(similarities)[::-1][1:21])  # Top 20, excluding self

                    # Include both: similar XLSX pages + relevant sitemap pages (same category)
                    source_category = parse_url_path(source_url)['category']

                    target_pages = []
                    for t in all_targets:
                        target_url = t['url']

                        # FILTER 1: No self-linking
                        if target_url == source_url:
                            continue

                        # FILTER 2: Only PLP URLs (no product detail pages)
                        if not is_plp_url(target_url):
                            continue

                        # FILTER 3: Semantic coherence (no beige->black, etc.)
                        if not urls_are_semantically_coherent(source_url, target_url):
                            continue

                        if t['source'] == 'xlsx' and t['idx'] in top_indices:
                            # XLSX target with high similarity
                            t_copy = t.copy()
                            t_copy['similarity'] = similarities[t['idx']]
                            target_pages.append(t_copy)
                        elif t['source'] == 'sitemap' and t['category'] == source_category:
                            # Sitemap target in same category
                            t_copy = t.copy()
                            t_copy['similarity'] = 0.5  # Default similarity for sitemap URLs
                            target_pages.append(t_copy)

                    # Sort by similarity * pagerank (with safe handling for sitemap URLs)
                    target_pages.sort(
                        key=lambda x: x.get('similarity', 0.5) * (1 + x['pagerank'] * 10),
                        reverse=True
                    )

                    # Limit to top 30 candidates for AI
                    target_pages = target_pages[:30]

                    if use_ai_suggestions and target_pages:
                        # Use AI to suggest intelligent link placements
                        suggestions = get_ai_link_suggestions(
                            source_text=extract_text_from_html(source_text),
                            source_url=source_url,
                            target_pages=target_pages,
                            api_key=api_key,
                            max_links=max_links,
                            model=ai_model
                        )

                        new_text, links_inserted = insert_ai_suggested_links(source_text, suggestions, source_url)

                        reasoning = '\n'.join([
                            f"• {l['anchor_text']} → {l['reason']}"
                            for l in links_inserted
                        ])

                    else:
                        # Fallback to keyword matching
                        new_text = source_text
                        links_inserted = []
                        reasoning = "Using keyword matching (AI disabled)"

                    results.append({
                        **row,
                        'Bottom SEO Text (with links)': new_text,
                        'Links Inserted': ' | '.join([
                            f"{l['anchor_text']} -> {l['target_url']}"
                            for l in links_inserted
                        ]),
                        'Links Count': len(links_inserted),
                        'AI Reasoning': reasoning
                    })

                    total_links += len(links_inserted)

                    # Update progress
                    progress = 50 + int((idx / len(df)) * 45)
                    progress_bar.progress(progress)
                    status.text(f"🔗 Processing page {idx + 1}/{len(df)}...")

                results_df = pd.DataFrame(results)
                st.session_state['results_df'] = results_df
                st.session_state['total_links'] = total_links

                progress_bar.progress(100)
                status.text("✅ Processing complete!")

                st.success(f"🎉 Done! Inserted {total_links} links across {len(df)} pages")
                st.balloons()

    with tab2:
        st.header("Analysis Results")

        if 'results_df' not in st.session_state:
            st.info("👆 Upload and process your data first")
        else:
            results_df = st.session_state['results_df']
            pagerank = st.session_state.get('pagerank', {})
            G = st.session_state.get('graph', nx.DiGraph())
            df = st.session_state.get('df', pd.DataFrame())

            # Metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Pages", len(results_df))
            with col2:
                pages_with_links = results_df[results_df['Links Count'] > 0].shape[0]
                st.metric("Pages with Links", pages_with_links)
            with col3:
                st.metric("Total Links Inserted", st.session_state.get('total_links', 0))
            with col4:
                avg_links = results_df['Links Count'].mean()
                st.metric("Avg Links/Page", f"{avg_links:.1f}")

            st.divider()

            # Show AI reasoning for a sample page
            if st.session_state.get('params', {}).get('use_ai'):
                st.subheader("🤖 AI Reasoning Examples")

                pages_with_reasoning = results_df[results_df['AI Reasoning'].str.len() > 10]
                if not pages_with_reasoning.empty:
                    sample_page = st.selectbox(
                        "Select a page to see AI reasoning",
                        pages_with_reasoning['URL'].tolist()[:20]
                    )
                    if sample_page:
                        reasoning = pages_with_reasoning[
                            pages_with_reasoning['URL'] == sample_page
                        ]['AI Reasoning'].iloc[0]
                        st.markdown(f"**Why these links were chosen:**\n\n{reasoning}")

            st.divider()

            # Charts
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("📊 PageRank Distribution")
                if pagerank:
                    pr_df = pd.DataFrame([
                        {'URL': str(df.iloc[idx].get('URL', '')), 'PageRank': score}
                        for idx, score in pagerank.items()
                    ]).sort_values('PageRank', ascending=False)

                    fig = go.Figure(data=[
                        go.Bar(
                            x=pr_df['URL'].head(15).apply(lambda x: x.split('/')[-1] if '/' in str(x) else str(x)[:20]),
                            y=pr_df['PageRank'].head(15),
                            marker_color='#1E88E5'
                        )
                    ])
                    fig.update_layout(
                        title="Top 15 Pages by PageRank",
                        xaxis_title="Page",
                        yaxis_title="PageRank Score",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("🔗 Links Distribution")
                links_df = results_df[['URL', 'Links Count']].sort_values('Links Count', ascending=False)
                fig = go.Figure(data=[
                    go.Bar(
                        x=links_df['URL'].head(15).apply(lambda x: x.split('/')[-1] if '/' in str(x) else str(x)[:20]),
                        y=links_df['Links Count'].head(15),
                        marker_color='#43A047'
                    )
                ])
                fig.update_layout(
                    title="Top 15 Pages by Links Inserted",
                    xaxis_title="Page",
                    yaxis_title="Links Count",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

            # Network graph
            st.subheader("🕸️ Link Network Visualization")
            if len(G.nodes()) > 0 and len(G.nodes()) < 150:
                fig = create_network_graph(G, df, pagerank)
                st.plotly_chart(fig, use_container_width=True)
            elif len(G.nodes()) >= 150:
                st.info("Network visualization disabled for large graphs (>150 nodes)")
            else:
                st.info("No network data available")

            # Results table
            st.subheader("📋 Detailed Results")
            display_cols = ['URL', 'Primary Keyword (Color + Type)', 'Links Count', 'Links Inserted']
            if 'AI Reasoning' in results_df.columns:
                display_cols.append('AI Reasoning')
            st.dataframe(
                results_df[[c for c in display_cols if c in results_df.columns]],
                use_container_width=True,
                height=400
            )

    with tab3:
        st.header("Export Results")

        if 'results_df' not in st.session_state:
            st.info("👆 Upload and process your data first")
        else:
            results_df = st.session_state['results_df']

            st.subheader("Download Options")

            col1, col2 = st.columns(2)

            with col1:
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    results_df.to_excel(writer, index=False, sheet_name='SEO Internal Links')

                st.download_button(
                    label="📥 Download Excel (XLSX)",
                    data=buffer.getvalue(),
                    file_name="seo_internal_links_output.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary"
                )

            with col2:
                json_output = results_df.to_json(orient='records', indent=2)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_output,
                    file_name="seo_internal_links_output.json",
                    mime="application/json"
                )

            st.divider()

            # Preview
            st.subheader("Preview Output")

            page_urls = results_df['URL'].tolist()
            selected_url = st.selectbox("Select a page to preview", page_urls)

            if selected_url:
                row = results_df[results_df['URL'] == selected_url].iloc[0]

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Original Text:**")
                    original = row.get('Bottom SEO Text', '')
                    if original and pd.notna(original):
                        st.code(str(original)[:2000] + ('...' if len(str(original)) > 2000 else ''), language='html')
                    else:
                        st.info("No original text")

                with col2:
                    st.markdown("**Text with Links:**")
                    with_links = row.get('Bottom SEO Text (with links)', '')
                    if with_links and pd.notna(with_links):
                        st.code(str(with_links)[:2000] + ('...' if len(str(with_links)) > 2000 else ''), language='html')
                    else:
                        st.info("No links inserted")

                if row.get('AI Reasoning'):
                    st.markdown("**🤖 AI Reasoning:**")
                    st.markdown(row['AI Reasoning'])

                if row.get('Links Inserted'):
                    st.markdown("**Links Inserted:**")
                    for link in str(row['Links Inserted']).split(' | '):
                        if link:
                            st.markdown(f"- {link}")


if __name__ == "__main__":
    main()
