def are_results_commutative(a_b: str, b_a: str) -> int:
    if a_b == b_a:
        return 0 if a_b != "draw" else 2
    if "draw" in (a_b, b_a):
        return 1
    return 2


def create_transcript_from_snippets(transcript_snippets: list[str]) -> str:
    return "\n\n".join(
        [
            f"Transcript Snippet {transcript_idx + 1}:\n\n{transcript}"
            for transcript_idx, transcript in enumerate(transcript_snippets)
        ]
    )
