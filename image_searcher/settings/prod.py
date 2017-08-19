from .common import *
import os
import pymysql
import json

from django.core.exceptions import ImproperlyConfigured

with open(os.path.join(BASE_DIR, "image_searcher/secret.json")) as f:
    secrets = json.loads(f.read())

# Keep secret keys in secrets.json
def get_secret(setting, secrets=secrets, default=None):
    try:
        if not secrets.get(setting) and default:
            return default
        return secrets[setting]
    except KeyError:
        error_msg = "Set the {0} environment variable".format(setting)
        raise ImproperlyConfigured(error_msg)

pymysql.install_as_MySQLdb()

ALLOWED_HOSTS = ['*']
DEBUG = True

DATABASES = {
    'default': {'ENGINE': os.environ.get('DB_ENGINE', 'django.db.backends.mysql'),
    'HOST': get_secret('DB_HOST'),
    'USER': get_secret('DB_USER'),
    'PASSWORD': get_secret('DB_PASSWORD'),
    'NAME': get_secret('DB_NAME'),
    'PORT': get_secret('DB_PORT'),
    },
    }

INSTALLED_APPS += ['storages']
# django-storages 앱 의존성 추가
# 기본 static/media 저장소를 django-storages로 변경
STATICFILES_STORAGE = 'bepo3.storages.StaticS3Boto3Storage'
DEFAULT_FILE_STORAGE = 'bepo3.storages.MediaS3Boto3Storage'
# S3 파일 관리에 필요한 최소 설정
# 소스코드에 설정정보를 남기지마세요. 환경변수를 통한 설정 추천
AWS_ACCESS_KEY_ID = get_secret('AWS_ACCESS_KEY_ID', secrets=secrets)
AWS_SECRET_ACCESS_KEY = get_secret('AWS_SECRET_ACCESS_KEY', secrets=secrets)
AWS_STORAGE_BUCKET_NAME = get_secret('AWS_STORAGE_BUCKET_NAME', secrets=secrets)
AWS_S3_REGION_NAME = get_secret('AWS_S3_REGION_NAME', secrets=secrets, default='ap-northeast-2')
