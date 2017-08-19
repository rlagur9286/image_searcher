from django.db import models
from django.core.urlresolvers import reverse
from django import forms
from django.conf import settings
from imagekit.models import ProcessedImageField
from imagekit.models import ImageSpecField
from imagekit.processors import Thumbnail


def min_length_3_validator(value):
    if len(value) < 3:
        raise forms.ValidationError('3글자 이상 입력해주세요')


class Post(models.Model):
    STATUS_CHOICES = (
        ('d', 'Draft'),
        ('p', 'Published'),
        ('w', 'Withdrawn'),
    )
    author = models.CharField(max_length=20)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, related_name='posts', blank=True)
    title = models.CharField(max_length=100, verbose_name='제목', validators=[min_length_3_validator],
                             help_text='포스팅 제목을 입력해주세요. 최대 100자 내외.')  # 길이 제한이 있는 문자열
    content = models.TextField(verbose_name='내용')  # 길이 제한이 없는 문자열

    tags = models.CharField(max_length=100, blank=True)
    status = models.CharField(max_length=1, choices=STATUS_CHOICES, blank=True)
    tag_set = models.ManyToManyField('Tag', blank=True)
    ip = models.CharField(max_length=15)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    # 원본 이미지와 썸네일 모두 저장
    # photo = models.ImageField(blank=True, upload_to='blog/post/%Y/%m/%d')
    # photo_thumbnail = ImageSpecField(
    #     source='photo',
    #     processors=[Thumbnail(300, 300)],
    #     format='JPEG',
    #     options={'quality': 60})
    # 원본 이미지는 저장하지 않고 썸네일로 바로 저장
    photo = ProcessedImageField(blank=True, upload_to='blog/post/%Y/%m/%d',
                                processors=[Thumbnail(100, 50)],
                                format='JPEG',
                                options={'quality': 60})

    class Meta:
        ordering = ['-id']

    def __str__(self):
        return self.title

    def get_absolute_url(self):
        return reverse('blog:post_detail', args=[self.id])


class Tag(models.Model):
    name = models.CharField(max_length=50, unique=True)

    def __str__(self):
        return self.name


class Comment(models.Model):
    post = models.ForeignKey(Post)  # 1:M 실제로는 post_id 가 생김
    message = models.TextField()
    author = models.CharField(max_length=20)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
