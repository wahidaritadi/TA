from django.db import models

# Create your models here.
class Hmsp(models.Model):
    date = models.DateField()
    open_price = models.FloatField()
    high_price = models.FloatField()
    low_price = models.FloatField()
    close_price = models.FloatField()
    volume = models.BigIntegerField()

    def __str__(self):
        return f"{self.date} - {self.close_price}"

class Prediction(models.Model):
    predictionRef = models.IntegerField()
    date = models.DateField()
    predicted_close_price = models.FloatField()

    class Meta:
        db_table = 'ms_prediction'

    def __str__(self):
        return f"{self.predictionRef} - {self.date}: {self.predicted_close_price}"

class LGPrediction(models.Model):
    predictionRef = models.IntegerField()
    predictionDateFrom = models.DateField()
    predictionDateTo = models.DateField()
    inputTime = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'lg_prediction'

    def __str__(self):
        return f"{self.predictionRef} from {self.predictionDateFrom} to {self.predictionDateTo} at {self.inputTime}"
