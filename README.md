# Patent Claim Search Engine 

[Click here to watch the 2-minute demo](https://drive.google.com/file/d/1pI9XfK1S39tWiVsNsW1mUeD3Ned35tBK/view?usp=sharing)

## Problem Statement 
Patent professionals often struggle to efficiently find relevant and related patent claims, which are the legally enforceable core of a patent. Traditional keyword search tools are limited — they often miss semantically similar language or fail to support domain-specific filters like classification codes or technical terminology in abstracts and titles. This makes it difficult to identify prior art or overlapping inventions, especially when language varies but concepts align.

This project builds a search engine that combines semantic understanding with metadata-based filtering to help users find relevant patent claims more accurately. It supports:
* Natural language queries (semantic search via OpenAI embeddings)
* Optional filters for classification codes, keywords, and exact titles
* Two-phase reranking using a cross-encoder to refine the top results

For my enhancements I implemented hybrid search and two phase ranking. I chose these because they directly reflect how real patent agents work: starting with a broad conceptual idea, then filtering for domain relevance, and finally needing fine-grained prioritization among candidates.

### Part 1 : Core search function 
* Loads and indexes patent claim data from JSON files 
  - Note: Patents with missing fields are included as long as they are not missing the claim field. For the purpose of filtering, a missing field is treated the same as a nonmatching field. 
* Uses OpenAI’s text-embedding-3-small to embed query and patent claims (rest of patent data is stored as metadata for filtering)
* Performs semantic search with ChromaDB and retrieves top-75 matches

### Part 2 : Hybrid search 
* Supports filtering by:
    * Classification code
    * Keywords in title/abstract
    * Exact title
* Filters can be strict (exclusion) or soft (score boosting) 
* On efficiency: 
  - Time without hybrid search averaged 0.712 s while time with hybrid searching averaged 0.719 s (over 10 trials, tested before adding two phase searching)
  - While this time difference is not huge for this relatively smaller dataset, it would be problematic for a larger dataset. In this prototype, I implemented hybrid search by running full semantic retrieval followed by symbolic post-filtering. This post-filtering strategy is not optimal. In a production setting, I would filter using SQL first and only semantically search relevant claims — either by dynamically building a Chroma index on filtered results or switching to a database like FAISS or Qdrant that supports hybrid search natively.

### Part 2 : Two-phase searching
* Implements reranking of semantic results using the cross-encoder/ms-marco-MiniLM-L-6-v2 model
* Final score = cross-encoder relevance × filters preference weight 
* The cross-encoder reranking phase is accurate but latency-sensitive. To optimize it, I would limit the number of candidates passed from the semantic search, use a smaller or batched cross-encoder model, and consider caching results for frequent queries. I would also explore early pruning, where low-confidence semantic candidates (e.g., similarity score below 0.75) are excluded from reranking entirely.

## To Run Locally 
1. Clone the repo 
```bash
git clone https://github.com/sbaidya1/thinkstruct_patent_search.git
cd thinkstruct_patent_search
```
2. Create a virutal environemt (optional but reccomended) 
```bash
python -m venv env
source env/bin/activate  # or `env\Scripts\activate` on Windows
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Create a `.env` file using `.env.example`, and add your own OpenAI API key
5. Build the claims index 
```bash
python -m app.scripts.create_index
```
6. Run the app and view it in http://127.0.0.1:5000/
```bash
python run.py
```
