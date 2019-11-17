import os

db_config = {
    'user': 'root',
    'password': os.environ['DB_PASS'],
    'host': 'db',
    'port': '3306',
    'database': 'sistema_acceso'
}
