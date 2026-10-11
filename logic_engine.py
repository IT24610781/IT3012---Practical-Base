class KnowledgeBase:
    def __init__(self):
        # A set prevents duplicate facts.
        self.facts = set()

        # Each rule contains a list of premises and one conclusion.
        self.rules = []

    def tell_fact(self, fact_string):
        """Add a known fact to the knowledge base."""
        self.facts.add(fact_string)

    def tell_rule(self, premise_list, conclusion_string):
        """Store a rule: all premises imply the conclusion."""
        self.rules.append(
            (list(premise_list), conclusion_string)
        )

    def clear_facts(self):
        """Clear current facts while keeping the rules."""
        self.facts.clear()

    def forward_chain(self):
        """Apply rules repeatedly until no new facts are found."""
        new_facts_added = True

        while new_facts_added:
            new_facts_added = False

            for premises, conclusion in self.rules:
                # Only infer conclusions that are not already known.
                if conclusion not in self.facts:

                    # Every premise must be known before the rule fires.
                    if all(
                        premise in self.facts
                        for premise in premises
                    ):
                        self.facts.add(conclusion)
                        new_facts_added = True