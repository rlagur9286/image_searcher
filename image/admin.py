from django.contrib import admin
from .models import Image


@admin.register(Image)
class ImageAdmin(admin.ModelAdmin):
    list_display = ['id', 'image_path', 'status', 'abs_size', 'label_name', 'tags'] # display 리스트 지정
    actions = ['make_published', 'make_draft']  # 액션 리스트 지정

    def abs_size(self, image):
        return '{} 글자'.format(len(image.abs))
    abs_size.short_description = 'SIZE' # 보여질 이름 재정의

    def make_published(self, request, queryset):    # 액션 정의 requst와 querset을 인자로 받음
        updated_count = queryset.update(status='p')
        self.message_user(request, '{}건의 포스팅을 Published 상태로 변경'.format(updated_count))
    make_published.short_description = '지정 포스팅을 Published 상태로 변경'

    def make_draft(self, request, queryset):    # 액션 정의 requst와 querset을 인자로 받음
        updated_count = queryset.update(status='d')
        self.message_user(request, '{}건의 포스팅을 Draft 상태로 변경'.format(updated_count))
    make_draft.short_description = '지정 포스팅을 Draft 상태로 변경'

# admin.site.register(Image)
