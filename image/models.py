from django.db import models
from django.core.urlresolvers import reverse


class Label(models.Model):
    label_name = models.CharField(max_length=255, unique=True)
    upload_time = models.DateTimeField(auto_now_add=True)
    description = models.CharField(max_length=100, blank=True)
    model = models.CharField(max_length=255, blank=True)

    def __str__(self):
        return self.label_name

    class Meta:
        ordering = ['id']   # 오름차순
        # ordering = ['-id']   # 내림차순

    def get_absolute_url(self):
        return reverse('image:list_image', args=[self.id])


class Image(models.Model):
    #label_name_id
    label = models.ForeignKey(Label, on_delete=models.CASCADE)  # 1:N
    image_name = models.CharField(max_length=255, unique=True)
    upload_time = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.image_name

    class Meta:
        ordering = ['id']   # 오름차순
        # ordering = ['-id']   # 내림차순

    def get_absolute_url(self):
        return reverse('image:list_image', args=[self.id])


