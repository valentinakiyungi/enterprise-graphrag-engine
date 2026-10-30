"""Structured-output schemas shared by ingestion and the query workflow."""

from typing import List

from pydantic import BaseModel, Field


class Triple(BaseModel):
  subject: str = Field(description="Canonical full name of the subject entity")
  predicate: str = Field(description="Short verb phrase, e.g. 'reports to', 'leads', 'depends on'")
  object: str = Field(description="Canonical full name of the object entity")


class TripleList(BaseModel):
  triples: List[Triple] = Field(default_factory=list)


class EntityList(BaseModel):
  entities: List[str] = Field(
      default_factory=list,
      description="Named entities (people, teams, projects, systems, products, policies) mentioned in the question",
  )
