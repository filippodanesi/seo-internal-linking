# 🔗 SEO Internal Linking Tool

Semantic analysis + PageRank simulation for optimal internal link placement.

## Features

- **Semantic Analysis**: Uses OpenAI embeddings to find contextually relevant linking opportunities
- **PageRank Simulation**: Calculates link equity distribution with configurable damping factor (0.85)
- **Smart Link Insertion**: Automatically inserts links in Bottom SEO Text with exact keyword matching
- **Interactive Visualization**: Network graph showing link structure and PageRank distribution
- **Export**: Download results as XLSX or JSON

## How it works

1. **Upload** your SEO keyword analysis Excel file
2. **Configure** parameters (max links, similarity threshold, etc.)
3. **Process** to generate semantic embeddings and calculate PageRank
4. **Review** the analysis and link suggestions
5. **Export** the updated content with internal links

## Required Excel Columns

- `URL`: Page URL
- `Primary Keyword (Color + Type)`: Main target keyword
- `Related Keyword`: Comma-separated related keywords
- `Bottom SEO Text`: HTML content to add links to

## Optional Columns

- `Title`: Page title (used for semantic analysis)
- `Meta Description`: Meta description

## Deploy on Streamlit Cloud

1. Fork/clone this repository
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy the app
4. Add your OpenAI API key in the app sidebar

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| Max links per page | 5 | Maximum internal links to insert |
| Min similarity | 0.5 | Minimum semantic similarity threshold |
| Damping factor | 0.85 | PageRank damping factor |
| Embedding model | text-embedding-3-small | OpenAI embedding model |

## The SEO Theory

Internal links are a **PageRank distribution system**. When a page receives external backlinks, that authority cascades through your site via internal links, diluting at each hop based on the damping factor (0.85).

Key insights:
- **Level 1** gets 85% of the source value
- **Level 2** gets ~72% (85% × 85%)
- By **Level 5**, only ~16% remains

This tool helps you:
1. Identify semantically relevant linking opportunities
2. Understand how authority flows through your site
3. Optimize link placement for maximum equity distribution

## License

MIT
