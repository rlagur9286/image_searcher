from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
from django.shortcuts import render

from .serializers import PostSerializer
from .models import Post
from .models import Comment

@api_view(['GET', 'POST'])
def post_list(request):
    if request.method == 'GET':
        post = Post.objects.all()
        serializer = PostSerializer(post, many=True)
        return Response(serializer.data)

    elif request.mothod == 'POST':
        serializer = PostSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET', 'PUT', 'DELETE'])
def post_detail(request, id):
    try:
        post = Post.objects.get(id=id)
    except Post.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)

    if request.method == 'GET':
        serializer = PostSerializer(post)
        return Response(serializer.data)

    elif request.method == 'PUT':
        serializer = PostSerializer(post, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    elif request.method == 'DELETE':
        post.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


def comment_list(request):
    comment_list = Comment.objects.all()
    return render(request, 'blog/comment_list.html', {'comment_list': comment_list})


# def post_list(request):
#     qs = Post.objects.all()
#     q = request.GET.get('q', '')
#     if q:
#         qs = qs.filter(title__icontains=q)
#     return render(request, 'blog/post_list.html', {'post_list': qs, 'q': q, })
#
#
# def post_new(request):
#     if request.method == 'POST':
#         form = PostModelForm(request.POST, request.FILES)
#         if form.is_valid():
#             post = form.save(commit=False)  # 직접 저장은 안하고 천천히 lazy 저장할게
#             post.ip = request.META['REMOTE_ADDR']
#             post.save()
#             # 방법 1
#             """
#             post = Post()
#             post.title = form.cleaned_data['title']
#             post.content = form.cleaned_data['content']
#             post.save()
#             """
#             # 방법 2
#             """
#             post = Post(title = form.cleaned_data['title'], content = form.cleaned_data['content'])
#             post.save()
#             """
#             # 방법 3
#             """
#             post = Post.create(title=form.cleaned_data['title'], content=form.cleaned_data['content'])
#             """
#
#             return redirect('blog:post_list')
#     else:
#         form = PostModelForm()
#     return render(request, 'blog/post_form.html', {'form': form, })
#
#
# def post_edit(request, id):
#     post = get_object_or_404(Post, id=id)
#
#     if request.method == 'POST':
#         form = PostModelForm(request.POST, request.FILES, instance=post)
#         if form.is_valid():
#             post = form.save(commit=False)
#             post.user = request.user
#             post.ip = request.META['REMOTE_ADDR']
#             post.save()
#             return redirect(post)   # post.get_absolute_url 로 이동함
#     else:
#         form = PostModelForm(instance=post)
#     return render(request, 'blog/post_form.html', {'form': form, })

