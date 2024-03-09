from django.shortcuts import render

def pdf_uploader(request):
    return render(request, 'index.html')

def embeddings(request):
    return render(request, 'embeddings.html')

def query(request):
    return render(request, 'query.html')
