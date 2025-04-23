import sqlite3
import os
from datetime import datetime

# Database file
DB_FILE = "visitor_logs.db"

def init_db():
    """Initialize the database and create the visitors table if not exists."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS visitor_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            category TEXT,
            time_of_visit TEXT,
            visit_count INTEGER DEFAULT 1
        )
    ''')
    conn.commit()
    conn.close()

def log_visit(name, category):
    """Log a visitor entry in the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Check if the visitor already exists in the logs today
    cursor.execute('''
        SELECT id, visit_count FROM visitor_logs 
        WHERE name = ? AND category = ? AND DATE(time_of_visit) = DATE(?)
    ''', (name, category, timestamp))
    existing_record = cursor.fetchone()
    
    if existing_record:
        # If visitor exists, update visit count
        visit_id, visit_count = existing_record
        cursor.execute('''
            UPDATE visitor_logs 
            SET visit_count = visit_count + 1, time_of_visit = ?
            WHERE id = ?
        ''', (timestamp, visit_id))
    else:
        # Insert new visitor log
        cursor.execute('''
            INSERT INTO visitor_logs (name, category, time_of_visit, visit_count)
            VALUES (?, ?, ?, ?)
        ''', (name, category, timestamp, 1))
    
    conn.commit()
    conn.close()

def get_logs(name=None, category=None, start_date=None, end_date=None):
    """Retrieve visitor logs with optional filters."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    query = "SELECT id, name, category, time_of_visit, visit_count FROM visitor_logs WHERE 1=1"
    params = []
    
    if name:
        query += " AND name LIKE ?"
        params.append(f"%{name}%")
    
    if category:
        query += " AND category = ?"
        params.append(category)
    
    if start_date and end_date:
        query += " AND DATE(time_of_visit) BETWEEN DATE(?) AND DATE(?)"
        params.append(start_date)
        params.append(end_date)
    
    query += " ORDER BY time_of_visit DESC"
    
    cursor.execute(query, params)
    logs = cursor.fetchall()
    conn.close()
    return logs

