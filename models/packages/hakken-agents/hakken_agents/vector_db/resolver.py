from collections.abc import Callable

from langchain_core.documents import Document

from hakken_agents.vector_db.engine import VectorDBEngine


async def resolve_documents(
    documents: list[Document],
    db: VectorDBEngine,
    similarity_fn: Callable,
    set_new_doc_fn: Callable,
    k: int = 20,
) -> list[Document]:
    num_new_docs = 0
    for doc in documents:
        relevant_docs_with_score = await db.asimilarity_search_with_score(doc.page_content, k=k)
    doc_is_new = True
    for rel_doc, distance in relevant_docs_with_score:
        doc_exists = similarity_fn(doc, rel_doc, distance)
        if doc_exists:
            doc_is_new = False
            break
    if doc_is_new:
        num_new_docs += 1
        ids = await db.aadd_documents([doc])
        doc.id = ids[0]
        set_new_doc_fn(doc, ids[0])
    return num_new_docs
