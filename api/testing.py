from exa_py import Exa
from dotenv import load_dotenv
import os
import json

load_dotenv()

exa = Exa(api_key = os.getenv("EXA_AI_API_KEY"))

result = exa.search_and_contents(
  """
  Search for the official clinical guidelines, peer-reviewed studies, or professional medical documentation that verifies, and establishes the standard of care and diagnostic criteria for this medical report:
  
  Human lymphatic filariasis is a serious disease found in tropical areas. It is spread by mosquito vectors that pick up tiny worm larvae, called microfilariae, from the blood of infected people. Long-lived adult parasites inside the body produce these microfilariae. Scientists studied substances released by these adult parasites, especially from a worm called Brugia malayi. They found that the most common substance released is a protein called triose phosphate isomerase (TPI). TPI is an enzyme, which means it helps chemical reactions in the body. Our body's immune system makes antibodies against TPI. However, general antibodies did not stop TPI from working. When twenty-three specific antibodies against TPI were created, only two of them could block its activity. Giving TPI as a vaccine to animals did not create protective immunity or antibodies that could stop the enzyme. However, giving specific blocking antibodies to mice before they were infected with adult Brugia malayi worms led to a 60-70% decrease in microfilariae levels. It also reduced the production of eggs and microfilariae by the adult female worms. This reduced fertility was also linked to less activity from a type of immune cell called CD4+ T cells and a higher number of immune cells called macrophages at the infection site. This research shows that active TPI is very important for the transmission cycle of Brugia malayi parasites. Because of its important role, TPI is now considered a possible target for new treatments or vaccines (immunological and pharmacological intervention) to fight filarial infections.
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

print(json.dumps([vars(r) for r in result.results], indent=2))
