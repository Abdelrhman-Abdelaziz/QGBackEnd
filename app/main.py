from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from Questiongen import main
from Questiongen import distrctors
import random
app = FastAPI()
qg = main.QGen()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    context: str
    n_ques: int
    n_mcq: int


class QuestionResponse(BaseModel):
    questions: list


class sentenceQuestionRequest(BaseModel):
    sentence: str


class sentenceQuestionResponse(BaseModel):
    question: str
    mcq: list
    answer: str

class distractorsRequest(BaseModel):
    word: str
    n:int


class distractorsResponse(BaseModel):
    distractors:list


@app.get('/')
def index():
    return {'message': 'hello world', "Author": "Abdelrhman Abdelaziz"}


@app.post("/mcq", response_model=QuestionResponse)
def getquestions(request: QuestionRequest):
    context = request.context
    n_mcq = request.n_mcq
    n_ques = request.n_ques
    if n_ques <= 0 or n_mcq <= 0:
        return QuestionResponse(questions=[])
    ques = qg.predict_mcq(context, n_ques, n_mcq)
    return QuestionResponse(questions=ques)


@app.post("/fillBlank", response_model=sentenceQuestionResponse)
def getFillBlank(request: sentenceQuestionRequest):
    keywords = qg.get_nouns_multipartite(request.sentence)
    keywords = distrctors.filter_keywords2(keywords, qg.s2v)
    if len(keywords) == 0:
        raise HTTPException(status_code=400, detail="Can't Make question")
    random.shuffle(keywords)
    answer = keywords[0]
    question = request.sentence.replace(answer, " ....... ")
    mcq = distrctors.get_distractors(answer,question,qg.s2v,qg.s_t_model,30,0.2)[:4]

#     mcq.append(answer)
    random.shuffle(mcq)

    return sentenceQuestionResponse(question= question, mcq= mcq ,answer= answer.capitalize())


@app.post("/distractors", response_model=distractorsResponse)
def getDistractors(request: distractorsRequest):
    word = request.word
    n = request.n
    mcq = distrctors.get_distractors(answer,question,qg.s2v,qg.s_t_model,30,0.2)[:n]

    return distractorsResponse(distrctors=mcq)
