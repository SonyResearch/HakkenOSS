from datetime import datetime

from pydantic import BaseModel


class UserDBModel(BaseModel):
    email: str
    name: str
    created_at: datetime
    updated_at: datetime
    last_login_at: datetime | None = None
    id: str | None = None
