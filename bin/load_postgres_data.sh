#!/bin/bash

set -e

# get arguments
# if $@ is empty, set it to all of our db's
if [ -z "$@" ]; then
    set -- academic advising atis broker car_dealership derm_treatment ewallet geography restaurants scholar yelp
fi
# $@ is all arguments passed to the script
echo "Databases to init: $@"

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
until PGPASSWORD="${DBPASSWORD:-postgres}" psql -U "${DBUSER:-postgres}" -h "${DBHOST:-localhost}" -p "${DBPORT:-5432}" -c '\q' 2>/dev/null; do
    echo "PostgreSQL is unavailable - sleeping"
    sleep 1
done
echo "PostgreSQL is ready!"

# get each folder name in data/database_data
for db_name in "$@"; do
    echo "dropping and recreating database ${db_name}"
    # drop and recreate database
    PGPASSWORD="${DBPASSWORD:-postgres}" psql -U "${DBUSER:-postgres}" -h "${DBHOST:-localhost}" -p "${DBPORT:-5432}" -c "DROP DATABASE IF EXISTS ${db_name};"
    PGPASSWORD="${DBPASSWORD:-postgres}" psql -U "${DBUSER:-postgres}" -h "${DBHOST:-localhost}" -p "${DBPORT:-5432}" -c "CREATE DATABASE ${db_name};"
    echo "done dropping and recreating database ${db_name}"
    
    db_path="data/database_data/${db_name}/${db_name}.sql"
    if [ -f "${db_path}" ]; then
        echo "importing ${db_path} into database ${db_name}"
        PGPASSWORD="${DBPASSWORD:-postgres}" psql -U "${DBUSER:-postgres}" -h "${DBHOST:-localhost}" -p "${DBPORT:-5432}" -d "${db_name}" -f "${db_path}"
        echo "successfully imported ${db_path} into database ${db_name}"
    else
        echo "Warning: ${db_path} not found, skipping..."
    fi
done

echo "PostgreSQL database initialization complete!"
