# Generated by Django 2.2.16 on 2020-10-23 06:20

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('helpdesk', '0036_auto_20201019_0233'),
    ]

    operations = [
        migrations.AlterField(
            model_name='ticket',
            name='queue',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='helpdesk.Queue', verbose_name='Queue'),
        ),
    ]