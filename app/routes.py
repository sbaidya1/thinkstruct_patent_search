"""
Flask routes for handling the UI and performing patent claim searches.

Includes:
- Rendering the homepage with the search form
- Handling search form submissions
- Performing hybrid + semantic search using ChromaDB
- Passing results and input fields back to the template
"""

from flask import Blueprint, request, render_template, current_app
from .search import run_search

main = Blueprint("main", __name__)

# home page route; displays search form
@main.route("/")
def index():
  return render_template("index.html")

# search route; handles POST request with filters + query
@main.route("/search", methods=["POST"])
def search():
  # get main search query
  query = request.form.get("query", "").strip()
  if not query:
    return render_template("index.html", query=query, results=[])

  # get optional metadata filters and their respective modes
  classification = request.form.get("classification", "").strip()
  keyword = request.form.get("keyword", "").strip().lower()
  title = request.form.get("title", "").strip().lower()
  classification_mode = request.form.get("classification_mode", "required")
  keyword_mode = request.form.get("keyword_mode", "required")
  title_mode = request.form.get("title_mode", "required")

  persist_dir = current_app.config["CHROMA_PERSIST_DIR"]

  # perform search
  results = run_search(
    query=query,
    persist_dir=persist_dir,
    classification=classification,
    keyword=keyword,
    title=title,
    classification_mode=classification_mode,
    keyword_mode=keyword_mode,
    title_mode=title_mode
  )

  # render results and input fields back to template
  return render_template(
    "index.html",
    query=query,
    classification=classification,
    keyword=keyword,
    title=title,
    classification_mode=classification_mode,
    keyword_mode=keyword_mode,
    title_mode=title_mode,
    results=results
)
