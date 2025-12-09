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


# ============================================================================
# AI-POWERED LINK SUGGESTION
# ============================================================================

def get_ai_link_suggestions(
    source_text: str,
    source_url: str,
    target_pages: List[Dict],
    api_key: str,
    max_links: int = 5
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

AVAILABLE TARGET PAGES TO LINK TO:
{targets_summary}

YOUR TASK:
Analyze the source content and identify {max_links} optimal places to insert internal links. For each suggestion:

1. Find a natural phrase or word IN THE EXISTING TEXT that would make a good anchor text
2. The anchor text should be contextually relevant to the target page
3. Prefer longer, descriptive anchor texts (2-4 words) over single words when natural
4. Don't link generic words like "here", "click", "this", etc.
5. Ensure the link adds value for the reader
6. Distribute links throughout the content, not all in one paragraph
7. The anchor text MUST exist exactly as written in the source content

Return your suggestions as a JSON array with this format:
[
  {{
    "anchor_text": "exact text from source to make into link",
    "target_url": "URL to link to",
    "reason": "brief explanation of why this link is valuable"
  }}
]

Return ONLY the JSON array, no other text."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
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


def insert_ai_suggested_links(html_text: str, suggestions: List[Dict]) -> Tuple[str, List[Dict]]:
    """Insert links based on AI suggestions"""
    if not html_text or not suggestions:
        return html_text, []

    result = html_text
    links_inserted = []
    used_anchors = set()

    for suggestion in suggestions:
        anchor = suggestion.get('anchor_text', '')
        target_url = suggestion.get('target_url', '')
        reason = suggestion.get('reason', '')

        if not anchor or not target_url:
            continue

        # Skip if already used this anchor
        if anchor.lower() in used_anchors:
            continue

        # Check if anchor exists in text and is not already a link
        # Use case-insensitive search but preserve original case
        pattern = re.compile(
            r'(?<!<a[^>]*>)(?<!["\'])(' + re.escape(anchor) + r')(?!["\'])(?!</a>)',
            re.IGNORECASE
        )

        # Only replace first occurrence
        match = pattern.search(result)
        if match:
            original_text = match.group(1)
            replacement = f'<a href="{target_url}">{original_text}</a>'
            result = result[:match.start(1)] + replacement + result[match.end(1):]

            links_inserted.append({
                'anchor_text': original_text,
                'target_url': target_url,
                'reason': reason
            })
            used_anchors.add(anchor.lower())

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

    # Main content
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "📊 Analysis", "📥 Export"])

    with tab1:
        st.header("Upload your data")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("SEO Data (XLSX)")
            xlsx_file = st.file_uploader(
                "Upload your SEO keyword analysis file",
                type=['xlsx'],
                help="Excel file with columns: URL, Primary Keyword, Related Keyword, Bottom SEO Text"
            )

        with col2:
            st.subheader("How it works")
            if use_ai_suggestions:
                st.info("""
                **🤖 AI-Powered Mode (Active)**

                The AI will analyze each page's content and:
                1. Understand the semantic context
                2. Find natural anchor text opportunities
                3. Match them to relevant target pages
                4. Insert links that add value for readers

                This mimics how an expert SEO would approach internal linking.
                """)
            else:
                st.info("""
                **📊 Similarity Mode**

                Uses embeddings to find semantically similar pages
                and matches keywords for link insertion.
                """)

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

                # Prepare target pages list
                all_targets = []
                for idx, row in df.iterrows():
                    url = row.get('URL', '')
                    if pd.isna(url):
                        continue
                    all_targets.append({
                        'idx': idx,
                        'url': str(url),
                        'keyword': str(row.get('Primary Keyword (Color + Type)', '')),
                        'category': parse_url_path(url)['category'],
                        'pagerank': pagerank_scores.get(idx, 0)
                    })

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

                    # Get top similar pages as candidates
                    similarities = similarity_matrix[idx]
                    top_indices = np.argsort(similarities)[::-1][1:21]  # Top 20, excluding self

                    target_pages = [
                        t for t in all_targets
                        if t['idx'] in top_indices and t['url'] != source_url
                    ]

                    # Sort by similarity * pagerank
                    target_pages.sort(
                        key=lambda x: similarities[x['idx']] * (1 + x['pagerank'] * 10),
                        reverse=True
                    )

                    if use_ai_suggestions and target_pages:
                        # Use AI to suggest intelligent link placements
                        suggestions = get_ai_link_suggestions(
                            source_text=extract_text_from_html(source_text),
                            source_url=source_url,
                            target_pages=target_pages,
                            api_key=api_key,
                            max_links=max_links
                        )

                        new_text, links_inserted = insert_ai_suggested_links(source_text, suggestions)

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
