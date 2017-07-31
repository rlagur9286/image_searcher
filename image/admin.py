from django.contrib import admin
from .models import Image


@admin.register(Image)
class ImageAdmin(admin.ModelAdmin):
    list_display = ['id', 'image_path', 'abs_size', 'label_name', 'tags']

    def abs_size(self, image):
        return '{} 글자'.format(len(image.abs))
    abs_size.short_description = 'SIZE'
# admin.site.register(Image)
