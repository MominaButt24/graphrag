from src.agent.agent import run_agent
from src.config.logging import get_logger
from src.retrieval.graph import smart_query
from src.retrieval.hybrid import hybrid_answer
from src.services.errors import map_pipeline_error
from src.retrieval.query_enhancer import enhance_query
logger = get_logger(__name__)


def query_graph(question: str):
    logger.info(f"[/query] question='{question}'")
    try:
        return smart_query(question)
    except Exception as e:
        raise map_pipeline_error(e)


def query_hybrid(question: str):
    logger.info(f"[/query/hybrid] question='{question}'")
    try:
        # return hybrid_answer(question)
        retrieval_query = enhance_query(question)

        return hybrid_answer(
            question=question,
            retrieval_question=retrieval_query,
        )
    except Exception as e:
        raise map_pipeline_error(e)


def query_agent(question: str):
    logger.info(f"[/query/agent] question='{question}'")
    try:
        # return run_agent(question)
        retrieval_query = enhance_query(question)

        return run_agent(
            question,
            retrieval_query=retrieval_query,
        )
    except Exception as e:
        raise map_pipeline_error(e)
