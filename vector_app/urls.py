from django.urls import path
from .views import TextUploader, TextReader, QueryResponse
from .indexes import pdf_uploader, embeddings, query


urlpatterns = [
    path('', pdf_uploader, name='pdf-uploader'),
    path('embeddings/', embeddings, name='pdf-embeddings'),
    path('query/', query, name='query'),
    path('upload-text-file/', TextUploader.as_view(), name='text-uploader'),
    path('text-reader/', TextReader.as_view(), name='text-reader'),
    path('query-response/', QueryResponse.as_view(), name='query-response'),    
]
