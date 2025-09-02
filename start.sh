#!/bin/sh

# Exit on error
set -e

# Create the /data directory if it doesn't exist.
# This ensures the directory is available before the Python script runs.
mkdir -p /data

# Run a simple Python script inline to initialize the database
# This is more direct than calling a separate file.
python -c "
import os
from app import app, db, User

print('Release command started: Initializing database...')

with app.app_context():
    print('Ensuring database tables exist...')
    db.create_all()
    print('Tables created or already exist.')

    if User.query.first() is None:
        print('No users found. Seeding default users...')
        DEFAULT_USERS = [
            {'username': 'admin', 'password': 'admin'},
            {'username': 'SSA', 'password': 'Gay'},
            {'username': 'Ethos', 'password': 'Hasini'}
        ]
        for user_data in DEFAULT_USERS:
            print(f'Creating default user: {user_data[\"username\"]}')
            new_user = User(username=user_data['username'])
            new_user.set_password(user_data['password'])
            db.session.add(new_user)
        db.session.commit()
        print('Default users seeded.')
    else:
        print('Users already exist. Skipping seeding.')

print('Release command finished: Database is ready.')
"