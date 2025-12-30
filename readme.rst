🐍 Django Configuration Project
===============================

This repository provides a primitive yet production-ready configuration for a Django 4.2 application using **PostgreSQL**, **Redis**, **RabbitMQ**, **Celery**, **JWT Authentication**, and **DRF Spectacular** for API documentation. All services are containerized using **Docker Compose**.

---

🚀 Features
-----------

- Django 4.2
- PostgreSQL Database
- Redis Cache & Channels
- RabbitMQ + Celery for task queues
- JWT Authentication (via ``rest_framework_simplejwt``)
- API schema & docs using **DRF Spectacular** (Swagger & Redoc)
- Supports both WSGI and ASGI (for real-time features)
- Uses **Docker** and **Docker Compose** for easy development setup
- Static & media file handling ready
- Basic logging setup

---

🗂️ Project Structure
--------------------

.. code-block:: text

    src/
    ├── authentication/   # Custom User Model + JWT auth endpoints
    ├── core/             # Core app APIs
    ├── configuration/    # Django settings and routing
    ├── media/            # Media files
    └── static/           # Static files
    docker-compose.yml
    Dockerfile
    entrypoint.sh
    requirements.txt
    .env
    README.md

---

🐳 Running with Docker Compose
-----------------------------

1. Copy and edit ``.env``

.. code-block:: bash

    cp .env.example .env

Update credentials in ``.env`` (like ``SECRET_KEY``, ``DB_PASSWORD``, ``REDIS_PASSWORD``, ``RABBITMQ_PASSWORD``).

2. Build and run services

.. code-block:: bash

    docker-compose up --build

3. Visit API Docs

- Swagger UI: http://localhost:8000/api/schema/swagger-ui/
- Redoc: http://localhost:8000/api/schema/redoc/

---

⚙️ Services via Docker Compose
------------------------------

+------------+----------------------------------------+-----------------------------+
| Service    | Description                            | Port                        |
+============+========================================+=============================+
| db         | PostgreSQL database                    | ``${DB_PORT}:5432``          |
+------------+----------------------------------------+-----------------------------+
| redis      | Redis server (cache & channels)        | ``${REDIS_PORT_NUMBER}:6379``|
+------------+----------------------------------------+-----------------------------+
| rabbitmq   | RabbitMQ broker + Management UI        | ``${RABBITMQ_PORT_NUMBER}:5672``|
+------------+----------------------------------------+-----------------------------+
| api_local  | Django API running on Uvicorn (ASGI)   | ``${DJANGO_PORT}:8000``      |
+------------+----------------------------------------+-----------------------------+

---

Installed Packages
----------------------

**Core Django Apps**

- django
- django.contrib.admin
- django.contrib.auth
- django.contrib.sessions
- django.contrib.messages
- django.contrib.staticfiles

**Custom Apps**

- core
- authentication

**Third Party Apps**

- djangorestframework
- rest_framework_simplejwt
- drf_spectacular
- drf_spectacular_sidecar
- corsheaders
- channels
- channels_redis
- django_celery_results
- django_celery_beat
- django_extensions
- whitenoise (for static files)

---

🔐 Auth & Security
------------------

- Uses **JWT Authentication** via ``rest_framework_simplejwt``
- Custom user model (``AUTH_USER_MODEL = "authentication.User"``)
- Password validators enabled
- CORS and CSRF trusted origins ready to configure

---

📚 API Endpoints
----------------

+----------------------------------------+----------------------------+
| Endpoint                               | Description                |
+========================================+============================+
| ``/admin/``                             | Django Admin Panel         |
+----------------------------------------+----------------------------+
| ``/api/``                               | API Root                   |
+----------------------------------------+----------------------------+
| ``/api/schema/``                        | OpenAPI Schema             |
+----------------------------------------+----------------------------+
| ``/api/schema/swagger-ui/``             | Swagger Docs UI            |
+----------------------------------------+----------------------------+
| ``/api/schema/redoc/``                  | Redoc Docs UI              |
+----------------------------------------+----------------------------+
| ``/api/account/token/refresh``          | JWT Refresh Token          |
+----------------------------------------+----------------------------+

---

🔄 Celery & Task Queues
-----------------------

- Uses **RabbitMQ** as broker
- Results stored in Django DB
- Ready for periodic tasks using ``django_celery_beat``

Example Broker URL:

.. code-block:: env

    amqp://your_user:your_password@rabbitmq:5672//

---

💾 Database & Caching
----------------------

- **PostgreSQL** as the primary database
- **Redis** for caching & Django Channels (real-time features)
- Django ORM configured with environment variables

---

⚡ Running Management Commands
-----------------------------

For example, to create a superuser:

.. code-block:: bash

    docker-compose exec api_local python manage.py createsuperuser

---

📝 Logs
-------

- Logs output to ``stdout`` (console)
- Integrated logging for Django and Celery
- Optional Sentry integration ready in logger config (disabled by default)

---

Useful Commands
------------------

**Migrate Database**

.. code-block:: bash

    docker-compose exec api_local python manage.py migrate

**Collect Static Files**

.. code-block:: bash

    docker-compose exec api_local python manage.py collectstatic --noinput

**Open Django Shell**

.. code-block:: bash

    docker-compose exec api_local python manage.py shell

---

Requirements
----------------

- Python 3.12
- Docker & Docker Compose installed
- (Optional) PostgreSQL, Redis, RabbitMQ installed if running outside Docker

---

💪 Built With
-------------

- Django 4.2
- Django REST Framework
- Celery + RabbitMQ
- Redis
- DRF Spectacular (API docs)
- Docker

---

👤 Author
---------

Meisam Hakimi — `<https://github.com/MeisamHakimi>`_

---

📜 License
-----------

This project is licensed under the MIT License.

---

🤝 Contributing
----------------

Feel free to open issues or submit PRs if you'd like to improve this template project!
