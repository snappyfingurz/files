"""
example_run.py — Demonstrates 3 episodes of the Self-Improving Customer Support
Agent environment: one per difficulty level.

Run with:  python example_run.py
"""

from env import CustomerSupportEnv
from models import Action


def make_action(response: str, reflection: str) -> Action:
    return Action(response=response, reflection=reflection)


# ── Episode 1: Easy ──────────────────────────────────────────────────────────
def episode_easy(env: CustomerSupportEnv) -> None:
    print("\n" + "━" * 60)
    print("EPISODE 1 — easy_001")
    print("━" * 60)
    obs = env.reset(task_id="easy_001")
    print(f"Customer: {obs.customer_message[:120]}…")
    print(f"Past mistakes in memory: {obs.past_mistakes}")

    action = make_action(
        response=(
            "I'm truly sorry to hear your headphones haven't arrived yet — "
            "that's completely unacceptable and I fully understand your frustration. "
            "Let me pull up order #78342 right now and track exactly where it is. "
            "If the package is confirmed lost or significantly delayed, I will "
            "immediately arrange a replacement shipment or issue a full refund — "
            "whichever you prefer. I'll also create a case reference (#CS-2024-001) "
            "so you can track our progress. You'll hear back from me within "
            "2 business hours with a full update. Again, I sincerely apologize "
            "for the inconvenience."
        ),
        reflection=(
            "The customer is angry about a delayed shipment. I must apologize, "
            "reference the order number, commit to investigating, and offer a "
            "concrete resolution (refund or replacement). No forbidden phrases used."
        ),
    )

    result = env.step(action)
    print(f"Reward: {result.reward:.4f}  |  Done: {result.done}")
    print(f"Tone: {result.info['tone_score']:.2f}  "
          f"Correctness: {result.info['correctness_score']:.2f}  "
          f"Resolution: {result.info['resolution_score']:.2f}  "
          f"Action: {result.info['actionability_score']:.2f}  "
          f"Policy: {result.info['policy_compliance_score']:.2f}  "
          f"Concise: {result.info.get('conciseness_score', 0):.2f}  "
          f"Clear: {result.info.get('clarity_score', 0):.2f}")
    print(f"Mistakes: {result.info['mistakes_found'] or 'none'}")
    print(f"Feedback:\n{result.info['feedback']}")


# ── Episode 2: Medium ─────────────────────────────────────────────────────────
def episode_medium(env: CustomerSupportEnv) -> None:
    print("\n" + "━" * 60)
    print("EPISODE 2 — medium_001")
    print("━" * 60)
    obs = env.reset(task_id="medium_001")
    print(f"Customer: {obs.customer_message[:160]}…")
    print(f"Past mistakes carried into this episode: {obs.past_mistakes}")

    action = make_action(
        response=(
            "I'm sincerely sorry — three things going wrong at once is "
            "genuinely unacceptable, and I appreciate you bringing each of them "
            "to our attention. Let me address every issue right now.\n\n"
            "1. Double charge ($49.99 × 2): I can see both transactions and I'm "
            "initiating a full refund of $49.99 this moment. It will appear on "
            "your statement within 3–5 business days. Reference: REF-BILL-441.\n\n"
            "2. App crash on account settings: I'm logging this as a confirmed bug "
            "with our engineering team immediately. As a workaround, try clearing "
            "the app cache or using our web portal at account.example.com.\n\n"
            "3. Ticket #5512: Five days without a response is not acceptable. "
            "I'm escalating ticket #5512 to a supervisor right now and personally "
            "ensuring you receive a substantive reply within 24 hours.\n\n"
            "I understand you have every reason to be frustrated. We are resolving "
            "all three items today. Thank you for your patience."
        ),
        reflection=(
            "Three issues: billing, technical bug, and ignored ticket. I must "
            "address each explicitly, commit to refunding the duplicate charge, "
            "acknowledge the bug, and escalate the old ticket. Memory shows "
            "'missing_apology' from last time — I led with a strong apology this "
            "time to avoid that repeated mistake."
        ),
    )

    result = env.step(action)
    print(f"Reward: {result.reward:.4f}  |  Done: {result.done}")
    print(f"Tone: {result.info['tone_score']:.2f}  "
          f"Correctness: {result.info['correctness_score']:.2f}  "
          f"Resolution: {result.info['resolution_score']:.2f}")
    print(f"Improvement bonus: +{result.info['improvement_bonus']:.2f}  "
          f"Repeated penalty: -{result.info['repeated_mistake_penalty']:.2f}")
    print(f"Mistakes: {result.info['mistakes_found'] or 'none'}")
    print(f"Feedback:\n{result.info['feedback']}")


# ── Episode 3: Hard ───────────────────────────────────────────────────────────
def episode_hard(env: CustomerSupportEnv) -> None:
    print("\n" + "━" * 60)
    print("EPISODE 3 — hard_001")
    print("━" * 60)
    obs = env.reset(task_id="hard_001")
    print(f"Customer: {obs.customer_message[:160]}…")
    print(f"Past mistakes carried: {obs.past_mistakes}")

    action = make_action(
        response=(
            "I completely understand how frustrating this feels — you've been a "
            "loyal customer for nearly a year and it's natural to expect that "
            "to be recognised. I genuinely appreciate your business.\n\n"
            "I have to be transparent with you: our 30-day money-back guarantee "
            "is a firm policy window, and at 11 months of active use I'm not able "
            "to authorise a full refund outside of that period. I know that's "
            "not what you were hoping to hear, and I'm sorry.\n\n"
            "What I *can* offer as a concrete alternative:\n"
            "• A 90-day account pause — your data and settings are preserved and "
            "you can resume whenever you're ready.\n"
            "• A 20 % discount on your next annual renewal.\n\n"
            "I want to find a resolution that works for you within what I'm "
            "authorised to do. Which of these options would be most helpful?\n\n"
            "If you'd like to escalate, I'm happy to connect you with a senior "
            "account manager who can review your case."
        ),
        reflection=(
            "This is a policy-conflict scenario. I cannot promise a full refund "
            "or use the word 'fraud'. I must empathise, clearly state the policy, "
            "and offer the two permitted alternatives (account pause and renewal "
            "discount). I avoided all forbidden phrases from memory. My reflection "
            "from past episodes warned me not to be dismissive — I validated the "
            "customer's frustration before explaining the policy."
        ),
    )

    result = env.step(action)
    print(f"Reward: {result.reward:.4f}  |  Done: {result.done}")
    print(f"Tone: {result.info['tone_score']:.2f}  "
          f"Correctness: {result.info['correctness_score']:.2f}  "
          f"Resolution: {result.info['resolution_score']:.2f}")
    print(f"Improvement bonus: +{result.info['improvement_bonus']:.2f}  "
          f"Repeated penalty: -{result.info['repeated_mistake_penalty']:.2f}")
    print(f"Mistakes: {result.info['mistakes_found'] or 'none'}")
    print(f"Feedback:\n{result.info['feedback']}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    env = CustomerSupportEnv(seed=42)

    episode_easy(env)
    episode_medium(env)
    episode_hard(env)

    print("\n" + "═" * 60)
    print(env.render())

    final_state = env.state()
    print(f"\nFinal cumulative reward: {final_state.score:.4f}")
    print(f"Total steps: {final_state.step_count}")
    print(f"Episodes recorded: {len(final_state.history)}")


if __name__ == "__main__":
    main()
