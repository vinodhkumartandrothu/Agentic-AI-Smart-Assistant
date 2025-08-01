


from celery import Celery
import os
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)


os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")

app = Celery("core")

app.config_from_object("django.conf:settings", namespace="CELERY")

app.autodiscover_tasks()
