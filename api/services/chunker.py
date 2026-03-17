import medspacy
import json

class ChunkerService:
    def __init__(self):
        self.nlp = medspacy.load()
        self.nlp.add_pipe("medspacy_sectionizer")

    def chunk(self, text):
        doc = self.nlp(text)

        paragraphs = []
        for section in doc._.sections:
            start, end = section.body_span
            content = doc[start:end].text.strip()
            paragraph = {
                "title": section.category,
                "content": content
            }
            paragraphs.append(paragraph)

        json_output = json.dumps(paragraphs, indent=4)
        return json_output