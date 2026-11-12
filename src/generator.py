from groq import Groq

from src.config import GROQ_API_KEY, LLM_MODEL, NOT_FOUND_MESSAGE


class GenerationError(Exception):
    """LLM se jawab na mil sake."""


SYSTEM_PROMPT = f"""You are a strict document question-answering assistant.
Answer ONLY using the context provided below. Do not use outside knowledge.
If the context does not contain the answer, reply EXACTLY with:
"{NOT_FOUND_MESSAGE}"
Keep answers clear and concise. Mention page numbers like (Page 3) when useful."""


def build_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[Source: {c['doc_name']}, Page {c['page']}]\n{c['text']}" for c in chunks
    )


def generate_answer(query: str, chunks: list[dict]) -> str:
    # Context hi nahi mila to LLM ko bulane ki zaroorat nahi (hallucination se bachao)
    if not chunks:
        return NOT_FOUND_MESSAGE

    if not GROQ_API_KEY:
        raise GenerationError("GROQ_API_KEY .env mein nahi mili.")

    user_prompt = f"Context:\n{build_context(chunks)}\n\nQuestion: {query}"

    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise GenerationError(f"LLM error: {e}")