import sqlite3

conn = sqlite3.connect('file:C:/MLForge/forge.db?mode=ro', uri=True)
cursor = conn.cursor()
cursor.execute('SELECT name FROM sqlite_master WHERE type="table"')
tables = cursor.fetchall()
for t in tables:
    print('===', t[0], '===')
    cursor.execute('PRAGMA table_info(' + t[0] + ')')
    for col in cursor.fetchall():
        print('  ', col[1], ':', col[2])
    print()