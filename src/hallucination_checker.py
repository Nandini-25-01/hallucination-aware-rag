from transformers import pipeline

verifier = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=256
)

def check_hallucination(answer: str, context_chunks: list[str]) -> dict:
    context = "\n\n".join(context_chunks)

    prompt = f"""
Check whether the ANSWER is supported by the CONTEXT.

Answer with one of:
- supported
- partially supported
- not supported

Then give a confidence score between 0 and 1.

CONTEXT:
{context}

ANSWER:
{answer}
"""

    output = verifier(prompt, do_sample=False)[0]["generated_text"].lower()

    # Infer verdict
    if "not supported" in output:
        verdict = "NOT_SUPPORTED"
    elif "partially" in output:
        verdict = "PARTIALLY_SUPPORTED"
    elif "supported" in output:
        verdict = "SUPPORTED"
    else:
        verdict = "UNKNOWN"

    # Infer confidence
    confidence = 0.0
    for token in output.split():
        try:
            val = float(token)
            if 0.0 <= val <= 1.0:
                confidence = val
                break
        except:
            pass

    if verdict == "SUPPORTED":
        confidence = 0.85
    elif verdict == "PARTIALLY_SUPPORTED":
        confidence = 0.6
    elif verdict == "NOT_SUPPORTED":
        confidence = 0.2
    else:
        confidence = 0.4

    return {
        "verdict": verdict,
        "confidence": confidence,
        "raw_output": output
}
