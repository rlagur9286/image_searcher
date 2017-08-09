from django.shortcuts import render
from django.shortcuts import get_object_or_404
from .models import Post
from .forms import PostModelForm
from django.shortcuts import redirect


def post_list(request):
    qs = Post.objects.all()

    q = request.GET.get('q', '')
    if q:
        qs = qs.filter(title__icontains=q)
    return render(request, 'blog/post_list.html', {'post_list': qs, 'q': q, })


def post_detail(request, id):
    post = get_object_or_404(Post, id=id)

    return render(request, 'blog/post_detail.html', {'post': post})


def post_new(request):
    if request.method == 'POST':
        form = PostModelForm(request.POST, request.FILES)
        if form.is_valid():
            post = form.save(commit=False)  # 직접 저장은 안하고 천천히 lazy 저장할게
            post.ip = request.META['REMOTE_ADDR']
            post.save()
            # 방법 1
            """
            post = Post()
            post.title = form.cleaned_data['title']
            post.content = form.cleaned_data['content']
            post.save()
            """
            # 방법 2
            """
            post = Post(title = form.cleaned_data['title'], content = form.cleaned_data['content'])
            post.save()
            """
            # 방법 3
            """
            post = Post.create(title=form.cleaned_data['title'], content=form.cleaned_data['content'])
            """

            return redirect('blog:post_list')
    else:
        form = PostModelForm()
    return render(request, 'blog/post_form.html', {'form': form, })


def post_edit(request, id):
    post = get_object_or_404(Post, id=id)

    if request.method == 'POST':
        form = PostModelForm(request.POST, request.FILES, instance=post)
        if form.is_valid():
            post = form.save(commit=False)
            post.user = request.user
            post.ip = request.META['REMOTE_ADDR']
            post.save()
            return redirect(post)   # post.get_absolute_url 로 이동함
    else:
        form = PostModelForm(instance=post)
    return render(request, 'blog/post_form.html', {'form': form, })

