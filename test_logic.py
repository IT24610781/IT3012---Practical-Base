from logic_engine import KnowledgeBase


def test_forward_chaining():
    kb = KnowledgeBase()

    kb.tell_rule(
        ["TargetVisible", "HasDust"],
        "SafeToEngage"
    )

    kb.tell_rule(
        ["SafeToEngage", "BloodseekerMissing"],
        "Retreat"
    )

    # Test 1: Enough information to infer SafeToEngage,
    # but not enough to infer Retreat.
    kb.tell_fact("TargetVisible")
    kb.tell_fact("HasDust")

    kb.forward_chain()

    assert "SafeToEngage" in kb.facts
    assert "Retreat" not in kb.facts

    print("Test 1 passed: Safe engagement.")

    # Test 2: Clear the previous situation, retaining the rules.
    kb.clear_facts()

    kb.tell_fact("TargetVisible")
    kb.tell_fact("HasDust")
    kb.tell_fact("BloodseekerMissing")

    kb.forward_chain()

    assert "SafeToEngage" in kb.facts
    assert "Retreat" in kb.facts

    print("Test 2 passed: Retreat inferred.")

    # Extra check: Reverse the rule order to verify repeated passes.
    kb.rules.reverse()
    kb.clear_facts()

    kb.tell_fact("TargetVisible")
    kb.tell_fact("HasDust")
    kb.tell_fact("BloodseekerMissing")

    kb.forward_chain()

    assert "Retreat" in kb.facts

    print("Test 3 passed: Rule order does not prevent inference.")
    print("All logic engine tests passed!")


if __name__ == "__main__":
    test_forward_chaining()