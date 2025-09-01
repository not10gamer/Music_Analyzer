# init_db.py

from app import app, db, User

# List of default users
# IMPORTANT: Consider changing these default passwords!
DEFAULT_USERS = [
    {'username': 'admin', 'password': 'admin'},
    {'username': 'SSA', 'password': 'Gay'},
    {'username': 'Ethos', 'password': 'Hasini'}
]

with app.app_context():
    # Create the database tables if they don't exist
    print("Creating all database tables...")
    db.create_all()
    print("Tables created.")

    # Add default users if they aren't already in the database
    for user_data in DEFAULT_USERS:
        if not User.query.filter_by(username=user_data['username']).first():
            print(f"Creating default user: {user_data['username']}")
            new_user = User(username=user_data['username'])
            new_user.set_password(user_data['password'])
            db.session.add(new_user)
        else:
            print(f"User {user_data['username']} already exists.")

    # Commit all new users to the database
    db.session.commit()
    print("Database initialization complete.")