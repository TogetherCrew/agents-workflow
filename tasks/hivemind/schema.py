from pydantic import BaseModel


class DestinationModel(BaseModel):
    queue: str
    event: str


class RouteModel(BaseModel):
    source: str
    destination: DestinationModel | None = None


class QuestionModel(BaseModel):
    message: str
    filters: dict | None = None


class ResponseModel(BaseModel):
    message: str


class AMQPPayload(BaseModel):
    communityId: str
    route: RouteModel
    question: QuestionModel
    response: ResponseModel | None = None
    metadata: dict | None = None


class HTTPPayload(BaseModel):
    communityId: str
    question: QuestionModel
    response: ResponseModel | None = None
    taskId: str


class Payload(BaseModel):
    event: str
    date: str  # or datetime; depends on how you handle it
    content: AMQPPayload
