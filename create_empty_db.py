import os
import sys
from sqlalchemy import Column, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine

path = os.path.abspath('data/train.db')

Base = declarative_base()

class Comment(Base):
    __tablename__ = 'comments'

    id = Column(String, primary_key=True)
    text = Column(String())
    vector = Column(String())
    toxic = Column(Integer())
    severe_toxic = Column(Integer())
    obscene = Column(Integer())
    threat = Column(Integer())
    insult = Column(Integer())
    identity_hate = Column(Integer())


engine = create_engine('sqlite:///'+path)
Base.metadata.create_all(engine)
