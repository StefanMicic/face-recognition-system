import psycopg2
from loguru import logger as log

from config import config


def connect():
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        params = config()

        log.info('Connecting to the PostgreSQL database...')

        conn = psycopg2.connect(**params)

        # cur = conn.cursor()

        # cur.execute('SELECT * from users')
        #
        # row = cur.fetchone()
        # while row:
        #     print(row)
        #     row = cur.fetchone()
        #
        # cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    # finally:
    #     if conn is not None:
    #         conn.close()
    #         print('Database connection closed.')

    return conn


if __name__ == '__main__':
    connect()
