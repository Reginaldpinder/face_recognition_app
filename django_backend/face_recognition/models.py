from django.db import models
import pgvector

class FaceEmbedding(models.Model):
    person_name = models.CharField(max_length=255)
    embedding = pgvector.VectorField(dimensions=128)  # FaceNet outputs 128D embeddings
