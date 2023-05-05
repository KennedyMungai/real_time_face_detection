"""The entrypoint to the application"""
from fastapi import FastAPI


app = FastAPI()


@app.get("/", name="Home", description="The home endpoint for the backend application", tags=['Home'])
async def home() -> dict[str, str]:
    """The home endpoint for the backend application

    Returns:
        dict[str, str]: _description_
    """
    return {"Hello": "World"}
