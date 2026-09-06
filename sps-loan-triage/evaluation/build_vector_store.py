# evaluation/build_vector_store.py
# Rebuild the ChromaDB vector store from lending_policy.json from scratch.
# Run this after adding new policy clauses or when the vectorstore is stale.
#
# Usage:
#   cd sps-loan-triage
#   python evaluation/build_vector_store.py
#
# Prerequisites:
#   ollama pull nomic-embed-text   (or: ollama pull all-minilm)

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tools.vector_store import build_vector_store, VECTORSTORE_PATH, POLICY_STORE_PATH
import shutil

def main():
    print("=" * 60)
    print("  POLICY VECTOR STORE BUILDER")
    print("=" * 60)
    print(f"  Policy source: {POLICY_STORE_PATH}")
    print(f"  Vector store:  {VECTORSTORE_PATH}")
    print()

    # Wipe existing store for a clean rebuild
    if os.path.exists(VECTORSTORE_PATH):
        print("  Clearing existing vector store...")
        shutil.rmtree(VECTORSTORE_PATH)

    print("  Embedding policy clauses via Ollama (nomic-embed-text)...")
    success = build_vector_store()

    if success:
        print("\n  Vector store built successfully.")
        print("  Policy retrieval will now use semantic search.")
    else:
        print("\n  Build failed. Possible causes:")
        print("    - Ollama not running: start with 'ollama serve'")
        print("    - Embedding model not pulled: 'ollama pull nomic-embed-text'")
        print("    - ChromaDB not installed: 'pip install chromadb'")
        print("\n  Policy retrieval will fall back to keyword matching.")
    print("=" * 60)

if __name__ == "__main__":
    main()
