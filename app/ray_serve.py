from pathlib import Path

from dotenv import load_dotenv
from ray import serve

ENV_FILE = Path("/workspace/.env")
load_dotenv(ENV_FILE, override=True)

from app.main import create_app


@serve.deployment
@serve.ingress(create_app())
class GraphRAGDeployment:
    pass


app = GraphRAGDeployment.bind()