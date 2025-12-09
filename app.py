"""
SEO Internal Linking Tool
AI-powered semantic analysis for intelligent internal link placement
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import json
import re
from typing import List, Dict, Tuple, Optional
import networkx as nx
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

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
    Check if URL is a Product Listing Page (PLP) vs Product Detail Page (PDP).
    PLPs are category pages, PDPs are individual product pages.

    PDP indicators:
    - Ends with .html
    - Contains numeric product codes (e.g., /10005020-0003.html)
    - Has very long numeric sequences
    """
    if not url:
        return False

    url_lower = url.lower()

    # PDP: ends with .html
    if url_lower.endswith('.html'):
        return False

    # PDP: contains numeric product codes (8+ digits)
    if re.search(r'/\d{8,}', url):
        return False

    # PDP: ends with a numeric code pattern like /12345 or /12345-6789
    if re.search(r'/\d+-\d+\.html?$', url) or re.search(r'/\d{5,}$', url):
        return False

    return True


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

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

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

        if use_ai_suggestions:
            ai_model = st.selectbox(
                "AI Model",
                [
                    "gpt-5.1",
                    "gpt-5-mini",
                    "gpt-4o",
                    "gpt-4o-mini",
                ],
                index=0,
                help="GPT-5.1 is the most intelligent model. GPT-5-mini for faster/cheaper processing."
            )
        else:
            ai_model = "gpt-5.1"

    # Main content
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
