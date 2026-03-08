SPELL_CHECK_PROMPT = """Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.

User query: 
{query}
"""

REWRITE_PROMPT = """Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: 
{query}
"""

EXPAND_PROMPT = """Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms appended with search query, nothing else.

Examples:
- "scary bear movie" -> "scary terrifying horror grizzly bear movie film"
- "action movie with bear" -> "action thriller adventure movie film grizzly bear"
- "comedy with bear" -> "comedy funny humor lighthearted bear grizzly"

User query: 
{query}
"""

INDIVIDUAL_RERANK_PROMPT = """Rate how well this movie matches the search query.

Query: 
{query}

Movie Title: 
{title}

Movie Description:
{description}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.
"""

BATCH_RERANK_PROMPT = """Rank the movies listed below by relevance to the following search query. 
For each movie, you are provided with the movie ID, movie title, and movie description.

Query: 
{query}

Movies:
{doc_list_str}

Return ONLY the movie IDs in order of relevance (best match first). 
Return a valid JSON list, nothing else.
For example: [75, 12, 34, 2, 1]
"""

EVALUATE_PROMPT = """Rate how relevant each result is to this query on a 0-3 scale:

Query: 
{query}

Results: 
{formatted_results}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers out than 0, 1, 2, or 3.
Return ONLY the scores in the same order you were given the documents. 
Return a valid JSON list, nothing else. For example: [2, 0, 3, 2, 0, 1]
"""

AUGMENT_PROMPT = """Answer the question or provide information based on the provided documents.
This should be tailored to Netflix users. Netflix is a movie streaming service.

Query: 
{query}

Documents:
{docs}

Provide a comprehensive answer that addresses the query.
Return the answer in text format. 
"""

SUMMARIZE_PROMPT = """
Provide information useful to this query by synthesizing information from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Netflix users. Netflix is a movie streaming service.

Query: 
{query}

Search Results:
{results}

Provide a comprehensive 5 to 6 sentence answer that combines information from multiple sources.
Return the answer in text format. 
"""

CITATIONS_PROMPT = """Answer the question or provide information based on the provided documents.
This should be tailored to Netflix users. Netflix is a movie streaming service.
If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

Query: 
{query}

Documents:
{documents}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources using [1], [2], etc. format when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the documents, say "I don't have enough information"
- Be direct and informative

Return the answer in text format. 
"""

ANSWER_PROMPT = """Answer the following question based on the provided documents.

Question: 
{query}

Documents:
{context}

General instructions:
- Answer directly and concisely
- Use only information from the documents
- If the answer isn't in the documents, say "I don't have enough information"
- Cite sources when possible

Guidance on types of questions:
- Factual questions: Provide a direct answer
- Analytical questions: Compare and contrast information from the documents
- Opinion-based questions: Acknowledge subjectivity and provide a balanced view

Return answer in text format.
"""

IMAGE_REWRITE_PROMPT = """Given the included image and text query, rewrite the text query to improve search results from a movie database. 
Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary
"""