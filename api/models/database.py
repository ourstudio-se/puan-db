# database.py
from sqlalchemy import create_engine, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from api.settings import EnvironmentVariables

env = EnvironmentVariables()

engine = create_engine(env.POSTGRESQL_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime

class Database(Base):
    __tablename__ = 'databases'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    versions = relationship("Version", back_populates="database")

class Version(Base):
    __tablename__ = 'versions'

    hash = Column(String, primary_key=True, index=True)
    database_id = Column(Integer, ForeignKey('databases.id'), nullable=False)
    parent_hash = Column(String, ForeignKey('versions.hash'), nullable=True)
    message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    data = Column(LargeBinary)  # This is the new binary field

    database = relationship("Database", back_populates="versions")
