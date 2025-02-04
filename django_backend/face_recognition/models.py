from django.db import models
from pgvector.django import VectorField  

class FaceEmbedding(models.Model):
    person_name = models.CharField(max_length=255)
    embedding = VectorField(dimensions=128)  
