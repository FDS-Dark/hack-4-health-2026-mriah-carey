from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from exa_py import Exa
from dotenv import load_dotenv
import json
import os

load_dotenv()

class VerifierService:
    def __init__(self):
        # Load biomedical model for semantic similarity
        self.model = SentenceTransformer("emilyalsentzer/Bio_ClinicalBERT")

    def verify_similarity(self, original_text: str, simplified_text: str) -> float:
        """
        Compute semantic similarity between original and simplified documents.

        Args:
            original_text: The original medical document text
            simplified_text: The simplified version of the document

        Returns:
            Cosine similarity score between 0 and 1
        """
        # Generate embeddings for both texts
        embeddings = self.model.encode([original_text, simplified_text])

        # Compute cosine similarity
        similarity = cosine_similarity(
            [embeddings[0]],
            [embeddings[1]]
        )[0][0]

        return float(similarity)
    
class ExaVerifier:
    def __init__(self):
        self.exa = Exa(api_key=os.getenv("EXA_AI_API_KEY"))

    def search_sources(self, medical_summary):
        result = self.exa.search_and_contents(
        f"""
        Search for the official clinical guidelines, peer-reviewed studies, or professional medical documentation that verifies, and establishes the standard of care and diagnostic criteria for this medical report:
        {medical_summary}
        """,
        category = "research paper",
        extras = {
          "links": 10
        },
        highlights = True,
        num_results = 10,
        summary = True,
        type = "auto",
        user_location = "US",
        livecrawl = "fallback"
        )

        return json.dumps([vars(r) for r in result.results], indent=2)
