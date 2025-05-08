# templates.py
import random

class PromptTemplateManager:
    def __init__(self):
        self.templates = {
            "T1": self.simple_ordered,
            "T2": self.reverse_ordered,
            "T3": self.relevant_irrelevant, # This might need adjustment based on how irrelevant_examples are structured
            "T4": self.all_random,
            "T5": self.creative_narrative,
        }

    def _get_example_texts(self, example_dict):
        """Safely extracts English and Spanish texts from an example dictionary."""
        # Assuming the keys are 'english' and 'spanish' as per data loading.
        # Adjust if your actual keys are different (e.g., 'en', 'es', 'source', 'target')
        # for the training data examples.
        english_text = example_dict.get('english', 'N/A_ENG')
        spanish_text = example_dict.get('spanish', 'N/A_SPA')
        return english_text, spanish_text

    def simple_ordered(self, query, examples):
        """Present examples in original order."""
        return self._mk_prompt(query, examples)

    def reverse_ordered(self, query, examples):
        """Present examples in reverse order."""
        return self._mk_prompt(query, list(reversed(examples)))

    def relevant_irrelevant(self, query, examples, irrelevant_examples=None):
        """
        2 relevant (from examples), 1 irrelevant (from irrelevant_examples).
        If irrelevant_examples is None, will just use the first 3 from examples.
        """
        # This template might need more specific logic if irrelevant_examples
        # are also dicts and need key-based access.
        # For now, assuming 'examples' and 'irrelevant_examples' follow the same structure.
        rel = examples[:2]
        if irrelevant_examples and len(irrelevant_examples) > 0:
            irr = random.choice(irrelevant_examples) # Assuming irrelevant_examples are structured like 'examples'
            ordered = rel + [irr]
        else:
            ordered = examples[:3] # Takes the first 3 example dicts
        return self._mk_prompt(query, ordered)


    def all_random(self, query, examples):
        """Shuffle all examples."""
        exs = examples[:] if examples else [] # examples is already a list of dicts
        random.shuffle(exs)
        return self._mk_prompt(query, exs)

    def no_examples(self, query, examples=None): # examples argument is not used here
        """Zero-shot: no examples, just the query."""
        return self._mk_prompt(query, []) # Pass an empty list for examples

    def creative_narrative(self, query, examples):
        """Wrap examples as "case studies"."""
        if not examples:
            return f"Translate this text from English to Spanish:\n{query}"

        story_ex = []
        for ex_dict in examples[:3]: # Iterate through the first 3 example dictionaries
            src, tgt = self._get_example_texts(ex_dict)
            story_ex.append(f"Case Study:\nPatient report: {src}\nTranslation: {tgt}")

        joined_stories = '\n---\n'.join(story_ex)
        return f"Use these case studies to translate:\n{joined_stories}\n\nNow translate:\n{query}"

    def get(self, template_id, query, examples):
        """Get a prompt using the specified template."""
        func = self.templates.get(template_id)
        if func is None:
            print(f"Warning: Template ID '{template_id}' not found. Using default simple_ordered.")
            func = self.simple_ordered
        return func(query, examples)

    def _mk_prompt(self, query, examples):
        """Create a prompt with examples and the query."""
        if not examples: # examples is a list of example dictionaries
            return f"Translate this English medical text to Spanish:\n{query}"

        ex_strings = []
        for ex_dict in examples: # Iterate through each example dictionary
            # Use the helper to get texts or specific keys directly
            # english_text = ex_dict.get('english', '[no English text]')
            # spanish_text = ex_dict.get('spanish', '[no Spanish text]')
            english_text, spanish_text = self._get_example_texts(ex_dict)
            ex_strings.append(f"Ex:\nEN: {english_text}\nES: {spanish_text}")

        ex_str = "\n\n".join(ex_strings)
        return f"{ex_str}\n\nTranslate: {query}"