from django.contrib import admin
from .models import Post
from .models import Comment
from .models import Tag


@admin.register(Post)
class PostAdmin(admin.ModelAdmin):
    list_display = ['id', 'title', 'abs_size', 'content', 'created_at', 'updated_at']  # display 리스트 지정
    actions = ['make_published', 'make_draft']  # 액션 리스트 지정

    def abs_size(self, post):
        return '{} 글자'.format(len(post.content))
    abs_size.short_description = 'SIZE' # 보여질 이름 재정의

    def make_published(self, request, queryset):    # 액션 정의 requst와 querset을 인자로 받음
        updated_count = queryset.update(status='p')
        self.message_user(request, '{}건의 포스팅을 Published 상태로 변경'.format(updated_count))
    make_published.short_description = '지정 포스팅을 Published 상태로 변경'


@admin.register(Tag)
class TagAdmin(admin.ModelAdmin):
    list_display = ['name']


@admin.register(Comment)
class CommentAdmin(admin.ModelAdmin):
    pass
