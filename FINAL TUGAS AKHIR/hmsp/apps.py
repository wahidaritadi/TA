from django.apps import AppConfig


class HmspConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'hmsp'

    def ready(self):
        from .views import fetch_and_save_stock_data
        fetch_and_save_stock_data()