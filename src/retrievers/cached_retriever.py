import hashlib
from typing import Dict, List

from langchain_core.documents import Document


class CachedRetriever:
    def __init__(self, base_retriever, cache_size: int = 100):
        self.base_retriever = base_retriever
        self.cache_size = cache_size
        self.cache: Dict[str, List[Document]] = {}

    def _hash_query(self, query: str) -> str:
        return hashlib.md5(query.lower().strip().encode()).hexdigest()

    def invoke(self, query: str) -> List[Document]:
        query_hash = self._hash_query(query)

        # Check cache first
        if query_hash in self.cache:
            return self.cache[query_hash]

        # Cache miss - call actual retriever
        result = self.base_retriever.invoke(query)

        # Add to cache (remove oldest if full)
        if len(self.cache) >= self.cache_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[query_hash] = result
        return result
