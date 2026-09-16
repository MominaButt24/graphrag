from src.config.logging import get_logger
from src.generation.llm import get_llm

logger = get_logger(__name__)
llm = get_llm()


def enhance_query(question: str) -> str:
	"""Rewrite a user question into a focused retrieval query.

	Retrieval stays available when the enhancer fails: the original question
	is returned instead of turning query enhancement into a hard dependency.
	"""
	question = question.strip()
	if not question:
		return question

	prompt = f"""Rewrite the following question for knowledge-base retrieval.
Keep the original meaning and important names, terms, and constraints.
Return only one concise standalone search query. Do not answer the question.

Question: {question}"""

	try:
		response = llm.invoke(prompt)
		enhanced = response.content.strip()
		
		return enhanced or question
	except Exception:
		logger.exception("Query enhancement failed; using the original question")
		return question
