"""
SEO Internal Linking Tool
Semantic analysis + PageRank simulation for optimal internal link placement
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
    .stProgress .st-bo {
        background-color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def parse_url_path(url) -> Dict:
    """Extract category structure from URL"""
    try:
        # Handle NaN, None, or non-string values
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


def escape_regex(string: str) -> str:
    """Escape special regex characters"""
    return re.escape(string)


def extract_text_from_html(html: str) -> str:
    """Remove HTML tags and extract plain text"""
    if not html:
        return ""
    # Remove tags
    text = re.sub(r'<[^>]+>', ' ', html)
    # Clean whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ============================================================================
# EMBEDDING FUNCTIONS
# ============================================================================

@st.cache_data(show_spinner=False)
def get_embeddings_batch(texts: List[str], api_key: str, model: str = "text-embedding-3-small") -> np.ndarray:
    """Get embeddings for a batch of texts using OpenAI API"""
    client = OpenAI(api_key=api_key)

    # Filter empty texts
    valid_indices = [i for i, t in enumerate(texts) if t and len(t.strip()) > 0]
    valid_texts = [texts[i] for i in valid_indices]

    if not valid_texts:
        return np.zeros((len(texts), 1536))

    # Batch requests (OpenAI allows up to 2048 texts per request)
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
                     similarity_threshold: float = 0.7) -> nx.DiGraph:
    """Build a directed graph based on semantic similarity"""
    G = nx.DiGraph()

    # Add nodes
    for idx, row in df.iterrows():
        G.add_node(idx,
                   url=row['URL'],
                   title=row.get('Title', ''),
                   category=parse_url_path(row['URL'])['category'])

    # Add edges based on similarity
    n = len(df)
    for i in range(n):
        for j in range(n):
            if i != j and similarity_matrix[i, j] >= similarity_threshold:
                G.add_edge(i, j, weight=similarity_matrix[i, j])

    return G


def calculate_pagerank(G: nx.DiGraph, damping: float = 0.85) -> Dict[int, float]:
    """Calculate PageRank for all nodes"""
    if len(G.nodes()) == 0:
        return {}
    try:
        return nx.pagerank(G, alpha=damping)
    except:
        return {n: 1.0/len(G.nodes()) for n in G.nodes()}


def simulate_equity_flow(G: nx.DiGraph, source_node: int,
                         damping: float = 0.85, max_depth: int = 6) -> Dict[int, Dict]:
    """Simulate how link equity flows from a source page"""
    equity = {source_node: {'equity': 1.0, 'depth': 0}}
    visited = set()
    current_level = {source_node}

    for depth in range(1, max_depth + 1):
        next_level = set()
        for node in current_level:
            if node in visited:
                continue
            visited.add(node)

            successors = list(G.successors(node))
            if not successors:
                continue

            node_equity = equity[node]['equity']
            distributed = node_equity * damping / len(successors)

            for succ in successors:
                if succ not in equity:
                    equity[succ] = {'equity': 0, 'depth': depth}
                equity[succ]['equity'] += distributed
                next_level.add(succ)

        current_level = next_level
        if not current_level:
            break

    return equity


# ============================================================================
# LINK INSERTION FUNCTIONS
# ============================================================================

def find_best_link_opportunities(
    source_idx: int,
    df: pd.DataFrame,
    similarity_matrix: np.ndarray,
    pagerank_scores: Dict[int, float],
    max_links: int = 5,
    min_similarity: float = 0.5
) -> List[Dict]:
    """Find the best pages to link to from a source page"""
    opportunities = []
    source_url = df.iloc[source_idx]['URL']
    source_parsed = parse_url_path(source_url)

    for target_idx in range(len(df)):
        if target_idx == source_idx:
            continue

        similarity = similarity_matrix[source_idx, target_idx]
        if similarity < min_similarity:
            continue

        target_url = df.iloc[target_idx]['URL']
        target_parsed = parse_url_path(target_url)

        # Calculate combined score
        # Semantic similarity (0-1) + PageRank boost + category bonus
        semantic_score = similarity
        pagerank_score = pagerank_scores.get(target_idx, 0) * 10  # Scale up

        # Category relevance bonus
        category_bonus = 0
        if source_parsed['category'] == target_parsed['category']:
            category_bonus += 0.2
        if source_parsed['subcategory'] != target_parsed['subcategory']:
            category_bonus += 0.1  # Cross-linking bonus

        combined_score = semantic_score + pagerank_score + category_bonus

        opportunities.append({
            'target_idx': target_idx,
            'target_url': target_url,
            'similarity': similarity,
            'pagerank': pagerank_scores.get(target_idx, 0),
            'combined_score': combined_score,
            'primary_keyword': df.iloc[target_idx].get('Primary Keyword (Color + Type)', '')
        })

    # Sort by combined score
    opportunities.sort(key=lambda x: x['combined_score'], reverse=True)
    return opportunities[:max_links]


def insert_links_in_text(
    html_text: str,
    link_opportunities: List[Dict],
    df: pd.DataFrame
) -> Tuple[str, List[Dict]]:
    """Insert links into HTML text based on keyword matches"""
    if not html_text:
        return html_text, []

    result = html_text
    links_inserted = []
    linked_urls = set()

    for opp in link_opportunities:
        target_url = opp['target_url']
        if target_url in linked_urls:
            continue

        # Get keywords for this target
        target_row = df.iloc[opp['target_idx']]
        keywords = []

        # Primary keyword
        pk = target_row.get('Primary Keyword (Color + Type)', '')
        if pk:
            keywords.append(pk.lower().strip())

        # Related keywords (first 5)
        related = target_row.get('Related Keyword', '')
        if related:
            related_list = [k.strip().lower() for k in str(related).split(',')[:5]]
            keywords.extend(related_list)

        # Sort by length (longer first)
        keywords = sorted(set(keywords), key=len, reverse=True)

        # Try to find and replace keyword
        replaced = False
        for keyword in keywords:
            if not keyword or len(keyword) < 3:
                continue

            # Match in paragraph content, not inside existing links
            pattern = re.compile(
                r'(<p>)(.*?)(</p>)',
                re.IGNORECASE | re.DOTALL
            )

            def replace_in_p(match):
                nonlocal replaced
                if replaced:
                    return match.group(0)

                p_start, content, p_end = match.groups()

                # Don't match inside existing <a> tags
                parts = re.split(r'(<a[^>]*>.*?</a>)', content, flags=re.IGNORECASE)
                new_parts = []

                for part in parts:
                    if part.startswith('<a') or replaced:
                        new_parts.append(part)
                    else:
                        # Try to find keyword
                        kw_pattern = re.compile(
                            r'(?<![\\w-])' + re.escape(keyword) + r'(?![\\w-])',
                            re.IGNORECASE
                        )
                        if kw_pattern.search(part):
                            new_part = kw_pattern.sub(
                                lambda m: f'<a href="{target_url}">{m.group(0)}</a>',
                                part,
                                count=1
                            )
                            if new_part != part:
                                replaced = True
                            new_parts.append(new_part)
                        else:
                            new_parts.append(part)

                return p_start + ''.join(new_parts) + p_end

            result = pattern.sub(replace_in_p, result)

            if replaced:
                links_inserted.append({
                    'keyword': keyword,
                    'url': target_url,
                    'similarity': opp['similarity'],
                    'pagerank': opp['pagerank']
                })
                linked_urls.add(target_url)
                break

    return result, links_inserted


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_network_graph(
    G: nx.DiGraph,
    df: pd.DataFrame,
    pagerank_scores: Dict[int, float],
    selected_node: Optional[int] = None
) -> go.Figure:
    """Create an interactive network visualization"""

    # Get positions using spring layout
    pos = nx.spring_layout(G, k=2, iterations=50)

    # Create edge traces
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

    # Create node traces
    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_text = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        # Color based on PageRank
        pr = pagerank_scores.get(node, 0)
        node_colors.append(pr)

        # Size based on PageRank
        node_sizes.append(10 + pr * 500)

        # Hover text
        url = df.iloc[node]['URL']
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
    st.markdown('<p class="sub-header">Semantic analysis + PageRank simulation for optimal internal link placement</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # API Key
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Your OpenAI API key for generating embeddings"
        )

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

        embedding_model = st.selectbox(
            "Embedding model",
            ["text-embedding-3-small", "text-embedding-3-large"],
            help="Larger model = better quality but higher cost"
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
            st.subheader("Sitemap (Optional)")
            sitemap_file = st.file_uploader(
                "Upload sitemap URLs (JSON or TXT)",
                type=['json', 'txt'],
                help="Optional: List of all site URLs for more complete analysis"
            )

        if xlsx_file:
            # Load data
            df = pd.read_excel(xlsx_file)
            st.success(f"✅ Loaded {len(df)} rows from Excel file")

            # Show preview
            with st.expander("Preview data"):
                st.dataframe(df.head(10))

            # Required columns check
            required_cols = ['URL', 'Primary Keyword (Color + Type)', 'Bottom SEO Text']
            missing_cols = [c for c in required_cols if c not in df.columns]

            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                st.stop()

            # Process button
            if st.button("🚀 Process Data", type="primary", disabled=not api_key):
                if not api_key:
                    st.error("Please enter your OpenAI API key in the sidebar")
                    st.stop()

                # Store in session state
                st.session_state['df'] = df
                st.session_state['api_key'] = api_key
                st.session_state['params'] = {
                    'max_links': max_links,
                    'min_similarity': min_similarity,
                    'damping_factor': damping_factor,
                    'embedding_model': embedding_model
                }

                # Progress
                progress_bar = st.progress(0)
                status = st.empty()

                # Step 1: Prepare text for embeddings
                status.text("📝 Preparing content for analysis...")
                texts = []
                for idx, row in df.iterrows():
                    # Combine title + primary keyword + bottom SEO text
                    text_parts = []
                    if pd.notna(row.get('Title')):
                        text_parts.append(str(row['Title']))
                    if pd.notna(row.get('Primary Keyword (Color + Type)')):
                        text_parts.append(str(row['Primary Keyword (Color + Type)']))
                    if pd.notna(row.get('Bottom SEO Text')):
                        text_parts.append(extract_text_from_html(str(row['Bottom SEO Text']))[:1000])
                    texts.append(' '.join(text_parts))

                progress_bar.progress(20)

                # Step 2: Get embeddings
                status.text("🧠 Generating semantic embeddings...")
                try:
                    embeddings = get_embeddings_batch(texts, api_key, embedding_model)
                    st.session_state['embeddings'] = embeddings
                except Exception as e:
                    st.error(f"Error getting embeddings: {str(e)}")
                    st.stop()

                progress_bar.progress(50)

                # Step 3: Calculate similarity
                status.text("🔄 Calculating semantic similarity...")
                similarity_matrix = calculate_semantic_similarity(embeddings)
                st.session_state['similarity_matrix'] = similarity_matrix

                progress_bar.progress(60)

                # Step 4: Build graph and calculate PageRank
                status.text("📈 Building link graph and calculating PageRank...")
                G = build_link_graph(df, similarity_matrix, min_similarity)
                pagerank_scores = calculate_pagerank(G, damping_factor)
                st.session_state['graph'] = G
                st.session_state['pagerank'] = pagerank_scores

                progress_bar.progress(80)

                # Step 5: Generate link suggestions and insert links
                status.text("🔗 Generating link suggestions and inserting links...")
                results = []
                total_links = 0

                for idx in range(len(df)):
                    row = df.iloc[idx].to_dict()

                    # Find best link opportunities
                    opportunities = find_best_link_opportunities(
                        idx, df, similarity_matrix, pagerank_scores,
                        max_links=max_links, min_similarity=min_similarity
                    )

                    # Insert links
                    original_text = row.get('Bottom SEO Text', '')
                    if original_text and pd.notna(original_text):
                        new_text, links_inserted = insert_links_in_text(
                            str(original_text), opportunities, df
                        )
                        row['Bottom SEO Text (with links)'] = new_text
                        row['Links Inserted'] = ' | '.join([
                            f"{l['keyword']} -> {l['url']} (sim: {l['similarity']:.2f})"
                            for l in links_inserted
                        ])
                        row['Links Count'] = len(links_inserted)
                        total_links += len(links_inserted)
                    else:
                        row['Bottom SEO Text (with links)'] = ''
                        row['Links Inserted'] = ''
                        row['Links Count'] = 0

                    results.append(row)

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
            pagerank = st.session_state['pagerank']
            G = st.session_state['graph']
            df = st.session_state['df']

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

            # PageRank distribution
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("📊 PageRank Distribution")
                pr_df = pd.DataFrame([
                    {'URL': df.iloc[idx]['URL'], 'PageRank': score}
                    for idx, score in pagerank.items()
                ]).sort_values('PageRank', ascending=False)

                fig = go.Figure(data=[
                    go.Bar(
                        x=pr_df['URL'].head(20).apply(lambda x: x.split('/')[-1] if '/' in x else x),
                        y=pr_df['PageRank'].head(20),
                        marker_color='#1E88E5'
                    )
                ])
                fig.update_layout(
                    title="Top 20 Pages by PageRank",
                    xaxis_title="Page",
                    yaxis_title="PageRank Score",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("🔗 Links by Page")
                links_df = results_df[['URL', 'Links Count']].sort_values('Links Count', ascending=False)
                fig = go.Figure(data=[
                    go.Bar(
                        x=links_df['URL'].head(20).apply(lambda x: x.split('/')[-1] if '/' in x else x),
                        y=links_df['Links Count'].head(20),
                        marker_color='#43A047'
                    )
                ])
                fig.update_layout(
                    title="Top 20 Pages by Links Inserted",
                    xaxis_title="Page",
                    yaxis_title="Links Count",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

            # Network graph
            st.subheader("🕸️ Link Network Visualization")
            if len(G.nodes()) < 200:  # Only show for smaller graphs
                fig = create_network_graph(G, df, pagerank)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Network visualization disabled for large graphs (>200 nodes)")

            # Detailed results table
            st.subheader("📋 Detailed Results")
            display_cols = ['URL', 'Primary Keyword (Color + Type)', 'Links Count', 'Links Inserted']
            st.dataframe(
                results_df[display_cols],
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
                # Excel export
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
                # JSON export
                json_output = results_df.to_json(orient='records', indent=2)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_output,
                    file_name="seo_internal_links_output.json",
                    mime="application/json"
                )

            st.divider()

            # Preview output
            st.subheader("Preview Output")

            # Select a page to preview
            page_urls = results_df['URL'].tolist()
            selected_url = st.selectbox("Select a page to preview", page_urls)

            if selected_url:
                row = results_df[results_df['URL'] == selected_url].iloc[0]

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Original Text:**")
                    original = row.get('Bottom SEO Text', '')
                    if original and pd.notna(original):
                        st.code(original[:2000] + ('...' if len(str(original)) > 2000 else ''), language='html')
                    else:
                        st.info("No original text")

                with col2:
                    st.markdown("**Text with Links:**")
                    with_links = row.get('Bottom SEO Text (with links)', '')
                    if with_links and pd.notna(with_links):
                        st.code(with_links[:2000] + ('...' if len(str(with_links)) > 2000 else ''), language='html')
                    else:
                        st.info("No links inserted")

                # Show links inserted
                if row.get('Links Inserted'):
                    st.markdown("**Links Inserted:**")
                    for link in str(row['Links Inserted']).split(' | '):
                        if link:
                            st.markdown(f"- {link}")


if __name__ == "__main__":
    main()
