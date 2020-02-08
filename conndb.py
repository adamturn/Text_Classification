# dedicated to liam
# thank you, liam :')
import psycopg2


def parse_props(props_path):
    """Parses information needed to connect to db and returns dictionary.
    
    Args:
        props_path: path to Java properties file.
    """
    print("Parsing properties...")
    props = open(props_path).read().split("\n")
    delim = "="
    props = {kv.split(delim)[0]: kv.split(delim)[1] for kv in props if len(kv) >= 3}
    
    return props


def connect_db(props_path):
    """Attempts to connect to db.
    Args:
        props_path: path to Java properties file.

    Returns: psycopg2 cursor object
    """
    props = parse_props(props_path)
    print("Connecting to db...")
    conn = psycopg2.connect(
        host=props['db_host'],
        database=props['db_name'],
        port=props['db_port'],
        user=props['db_user'],
        password=props['db_password']
    )
    print("\t--Connection established!")
    
    return conn.cursor()


if __name__ == '__main__':
    connect_db()
    print('--fin.')
