from ray import serve
from starlette.requests import Request


# 1: Define a Ray Serve application.
@serve.deployment
class MyModelDeployment:
    def __init__(self) -> None:
        pass

    def __call__(self, request: Request) -> dict:
        return {}


app = MyModelDeployment.bind()
serve.run(app, route_prefix="/")
