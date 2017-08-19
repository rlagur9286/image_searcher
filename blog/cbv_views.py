from django.views.generic import ListView, CreateView, DetailView, UpdateView, DeleteView
from django.core.urlresolvers import reverse_lazy
from .models import Post
from .models import Post
from .serializers import PostSerializer
from rest_framework import generics


class PostList(generics.ListCreateAPIView):
    queryset = Post.objects.all()
    serializer_class = PostSerializer


class PostDetail(generics.RetrieveUpdateDestroyAPIView):
    queryset = Post.objects.all()
    serializer_class = PostSerializer
    lookup_url_kwarg = 'id'

# post_detail = DetailView.as_view(model=Post, pk_url_kwarg='id')
# post_list = ListView.as_view(model=Post, paginate_by=10)
# post_new = CreateView.as_view(model=Post)
# post_edit = UpdateView.as_view(model=Post, pk_url_kwarg='id', fields='__all__')
# # post_delete = DeleteView.as_view(model=Post, success_url='/blog/cbv/', pk_url_kwarg='id') # 밑에와 같은 코드! reverse_lazy
# post_delete = DeleteView.as_view(model=Post, success_url=reverse_lazy('blog:post_list'), pk_url_kwarg='id')
