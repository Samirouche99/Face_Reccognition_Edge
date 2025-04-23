import json
import os

# Path to the categories JSON file
CATEGORY_FILE = "categories.json"

def load_categories():
    """Load categories and their associated colors from a JSON file."""
    if not os.path.exists(CATEGORY_FILE):
        # Create a default categories file if it doesn't exist
        default_categories = {
            "Barred": "Red",
            "VIP": "Green",
            "Staff": "Blue"
        }
        with open(CATEGORY_FILE, "w") as f:
            json.dump(default_categories, f, indent=4)
    
    # Load categories from the JSON file
    with open(CATEGORY_FILE, "r") as f:
        categories = json.load(f)
    
    return categories

def get_logs(name=None, category=None, start_date=None, end_date=None):
    """Retrieve visitor logs with optional filters."""
    conn = sqlite3.connect("visitor_logs.db")
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

