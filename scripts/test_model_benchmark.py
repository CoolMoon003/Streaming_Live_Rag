import time
import requests
import sys


MODEL = sys.argv[1] if len(sys.argv) > 1 else "gemma3:4b"

BASE_URL = "http://localhost:11434"


PROMPTS = {
    "speed":
        "Answer in exactly one short sentence: What is 2 + 2?",

    "grounded":
        """You are a strict corpus-grounded assistant.

Answer ONLY using the evidence below.

Every factual claim must end with the exact citation shown.

Evidence:
[DOC_TRAVEL_POLICY_C002 §2. International Travel]
International travel requires prior approval from the appropriate manager.
Employees may claim eligible airfare and accommodation expenses according to the applicable travel limits.

Question:
What is required for international travel?

Answer concisely with the citation.""",

    "missing_fact":
        """You are a strict corpus-grounded assistant.

Answer ONLY using the evidence below.
Do not invent missing information.

Evidence:
[DOC_TRAVEL_POLICY_C002 §2. International Travel]
International travel requires prior approval from the appropriate manager.
Employees may claim eligible airfare and accommodation expenses according to the applicable travel limits.

Question:
What is the maximum international airfare amount an employee can claim?

If the evidence does not contain the answer, clearly state that the provided corpus does not contain enough evidence.

Answer concisely.""",

    "multi_fact":
        """You are a strict corpus-grounded assistant.

Use ONLY the provided evidence.
Every factual claim must have an exact citation.

Evidence:

[DOC_TRAVEL_POLICY_C002 §2. International Travel]
International travel requires prior approval from the appropriate manager.
Employees may claim eligible airfare and accommodation expenses according to the applicable travel limits.

[DOC_TRAVEL_POLICY_C003 §3. Booking Requirements]
Travel should normally be booked through the approved booking process.
Late bookings may require additional approval.

Question:
What are the approval and booking requirements for international travel?

Answer concisely with citations."""
}


def run(prompt):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 120,
        },
    }

    start = time.perf_counter()

    response = requests.post(
        f"{BASE_URL}/api/generate",
        json=payload,
        timeout=120,
    )

    response.raise_for_status()

    elapsed = (time.perf_counter() - start) * 1000

    data = response.json()

    return elapsed, data


print("=" * 70)
print(f"MODEL BENCHMARK: {MODEL}")
print("=" * 70)

# Warm-up
print("\nWARM-UP...")
warmup_start = time.perf_counter()

requests.post(
    f"{BASE_URL}/api/generate",
    json={
        "model": MODEL,
        "prompt": "Reply with: READY",
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 10,
        },
    },
    timeout=120,
)

warmup_time = (time.perf_counter() - warmup_start) * 1000

print(f"Warm-up: {warmup_time:.0f} ms")


for name, prompt in PROMPTS.items():

    print("\n" + "-" * 70)
    print(f"TEST: {name}")

    elapsed, data = run(prompt)

    print(f"Wall time: {elapsed:.0f} ms")

    print(f"Load duration: "
          f"{data.get('load_duration', 0) / 1e9:.3f} s")

    print(f"Prompt evaluation: "
          f"{data.get('prompt_eval_duration', 0) / 1e9:.3f} s")

    print(f"Generation: "
          f"{data.get('eval_duration', 0) / 1e9:.3f} s")

    print(f"Prompt tokens: "
          f"{data.get('prompt_eval_count', 0)}")

    print(f"Output tokens: "
          f"{data.get('eval_count', 0)}")

    eval_duration = data.get("eval_duration", 0)

    if eval_duration:
        tok_sec = (
            data.get("eval_count", 0)
            / (eval_duration / 1e9)
        )

        print(f"Generation speed: {tok_sec:.2f} tok/s")

    print("\nOUTPUT:")
    print(data.get("response", "").strip())


print("\n" + "=" * 70)
print("BENCHMARK COMPLETE")
print("=" * 70)