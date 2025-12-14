from transformers import pipeline

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=256
)
def generate_answer(question: str, context_chunks: list[str]) -> str:
    """
    Generates an answer using only retrieved context.
    """
    context = "\n\n".join(context_chunks)
    prompt = f"""
Answer the question using only the context below.
If the answer is not present, say "I don't know the answer to this."
Context:{context}
Question:{question}
Answer:
"""
    output = generator(prompt, do_sample=False)
    return output[0]["generated_text"].strip()
