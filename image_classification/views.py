from django.shortcuts import render
from .forms import ImageUploadForm

def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Handle image upload and classification here
            pass
    else:
        form = ImageUploadForm()
    return render(request, 'upload_image.html', {'form': form})
