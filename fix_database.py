import sqlite3

connection = sqlite3.connect("landinsight.db")

try:
    connection.execute(
        "ALTER TABLE users ADD COLUMN email TEXT"
    )
    connection.commit()
    print("email column added successfully")

except sqlite3.OperationalError as error:
    print("Database update:", error)

connection.close()