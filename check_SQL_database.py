import numpy as np
import psycopg2 as pg2

conn = pg2.connect(database='ua_to_en', user='postgres', password='password')
cur = conn.cursor()

cur.execute('SELECT * FROM ua_to_en')
# cur.fetchmany(10)
word = cur.fetchone()
print(word[:])

conn.close()