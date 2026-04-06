from pydantic import BaseModel


class Title(BaseModel):
    text: str


class Category(BaseModel):
    category: str
