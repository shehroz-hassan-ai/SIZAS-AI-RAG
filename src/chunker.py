from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pages(pages, doc_name: str, chunk_size: int, chunk_overlap: int) -> list[dict]:
    """Har page ko chunks mein todta hai, metadata ke saath."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for page in pages:
        for n, piece in enumerate(splitter.split_text(page.text)):
            chunks.append(
                {
                    "chunk_id": f"{doc_name}-p{page.page_number}-c{n}",
                    "doc_name": doc_name,
                    "page": page.page_number,
                    "text": piece,
                }
            )
    return chunks