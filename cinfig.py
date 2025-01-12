from app import app, db

# Add debug prints
print("App import successful")
print("DB import successful")

try:
    with app.app_context():
        print("Inside app context")
        db.create_all()
        print("DB tables created successfully")
except Exception as e:
    print(f"An error occurred: {e}")