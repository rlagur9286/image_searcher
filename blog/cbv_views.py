from django.views.generic import ListView, CreateView, DetailView, UpdateView
from .models import Post


post_detail = DetailView.as_view(model=Post, pk_url_kwarg='id')
post_list = ListView.as_view(model=Post, paginate_by=10)
post_new = CreateView.as_view(model=Post)
post_edit = UpdateView.as_view(model=Post, pk_url_kwarg='id', fields='__all__')
