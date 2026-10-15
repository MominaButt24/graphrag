import os
from pathlib import Path

from dotenv import dotenv_values
import ray
from ray import serve

# Load variables from the project's root .env file
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
env_vars = {
    key: value
    for key, value in dotenv_values(ENV_FILE).items()
    if value is not None
}

# Make the variables available to Ray workers
env_vars.update({
    key: value
    for key, value in os.environ.items()
    if key in env_vars
})

ray.init(
    runtime_env={"env_vars": env_vars},
    ignore_reinit_error=True,
)

from app.main import app as fastapi_app


@serve.deployment
@serve.ingress(fastapi_app)
class GraphRAGDeployment:
    pass


app = GraphRAGDeployment.bind()