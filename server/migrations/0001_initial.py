# Generated by Django 4.2.1 on 2023-06-24 07:46

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Client',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('ip_address', models.CharField(max_length=50, verbose_name='ip_address')),
                ('port', models.CharField(max_length=10, verbose_name='port')),
                ('created_date', models.DateField(auto_now_add=True, verbose_name='created_date')),
                ('name', models.CharField(max_length=50, verbose_name='name')),
            ],
        ),
    ]