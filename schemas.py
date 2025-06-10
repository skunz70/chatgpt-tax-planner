from pydantic import BaseModel, EmailStr

class UserAuth(BaseModel):
    email: EmailStr
    password: str

class UserOut(BaseModel):
    email: EmailStr
    id: str

class TokenSchema(BaseModel):
    access_token: str
    refresh_token: str

class SystemUser(BaseModel):
    email: str
