from rest_framework.response import Response
from rest_framework.views import APIView
from .forms import UploadFileForm
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from utils.engine import EmbeddingModelManager, LanguageModelManager, RetrieverManager, FileReader
import time
import gc
import os
import uuid

gc.collect()

BASE_DIR = settings.BASE_DIR
MEDIA_ROOT = settings.MEDIA_ROOT
CHUNK_SIZE = int(settings.CHUNK_SIZE)
CHUNK_OVERLAP = int(settings.CHUNK_OVERLAP)
save_pdf_path = settings.MEDIA_ROOT + 'pdf'

class TextUploader(APIView):
    def post(self, request, format=None):
        if request.method == "POST":
            error_information = "No error"
            file_url = None
            status = 200
            try:
                form = UploadFileForm(request.POST, request.FILES)
                print("form is valid")
                if form.is_valid():
                    file = request.FILES["file"]
                    fs = FileSystemStorage(location=save_pdf_path)   
                    filename = fs.save(file.name, file)
                    file_url = os.path.join(save_pdf_path, filename)
                    print(file_url)
                else:
                    form = UploadFileForm()
            except Exception as e:
                error_information = str(e)
                status = 400
        return Response({'status':status,'file_url': file_url, 'error_information': error_information}, status=status)

class TextReader(APIView):
    def post(self, request, format=None):
        if request.method == "POST":
            error_information = "No error"
            extracted_text = None
            status = 200
            vector_db = ""
            try:
                file_reader = FileReader()
                file_url = request.data.get("file_url")
                collection_name = "session_" + str(uuid.uuid4()) + "_collection"
                extracted_text = file_reader.file_reader(str(file_url), CHUNK_SIZE, CHUNK_OVERLAP)
                print(f"docs: {extracted_text}")
                model_manager = EmbeddingModelManager()
                model = model_manager.embed_model()
                vector_db = model_manager.vector_storing(extracted_text, model, collection_name)
                del model
                if vector_db is False:
                    error_information = "Error in storing embeddings"
                    status = 400
            except Exception as e:
                error_information = str(e)
                status = 400
        return Response({'status':status, 'collection_name': collection_name, 'error_information': error_information}, status=status)

class QueryResponse(APIView):
    def post(self, request, format=None):
        if request.method == "POST":
            error_information = "No error"
            status = 200
            response = ""
            try:
                query = request.data.get('text')
                collection_name = request.data.get('collection_name')
        
                model_manager = EmbeddingModelManager()
                model = model_manager.embed_model()
                
                retriever_manager = RetrieverManager()
                start = time.time()
                llm = LanguageModelManager.load_mistral_model(temperature=0.00)
                response = retriever_manager.get_qa_retriever_text(llm, model, collection_name, query)
                end = time.time()
                del llm, model
                print(f"Time taken: {end - start} in seconds")
                
            except Exception as e:
                error_information = str(e)
                status = 400
        return Response({'response':response}, status=status)