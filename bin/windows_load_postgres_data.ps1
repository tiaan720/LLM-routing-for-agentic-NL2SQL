# Ensure PostgreSQL container is running
Write-Host "Starting PostgreSQL with Docker Compose..."
docker-compose up -d postgres


# Windows PostgreSQL Data Loader (PowerShell)
# Save as windows_load_postgres_data.ps1 and run in PowerShell

# Print all .sql files found under data/database_data for debugging
$searchPath = "data/database_data"
$sqlFiles = Get-ChildItem -Path $searchPath -Recurse -Filter *.sql

if ($sqlFiles.Count -eq 0) {
    Write-Host "No .sql files found under $searchPath"
} else {
    Write-Host "Found .sql files:"
    foreach ($file in $sqlFiles) {
        Write-Host $file.FullName
    }
}

$databases = @("academic", "advising", "atis", "broker", "car_dealership", "derm_treatment", "ewallet", "geography", "restaurants", "scholar", "yelp")

Write-Host "Databases to init: $($databases -join ', ')"

foreach ($db in $databases) {
    Write-Host "Dropping and recreating database $db..."
    docker-compose exec postgres psql -U postgres -c "DROP DATABASE IF EXISTS $db;"
    docker-compose exec postgres psql -U postgres -c "CREATE DATABASE $db;"
    Write-Host "done dropping and recreating database $db"

    $db_path = Join-Path -Path "data/database_data" -ChildPath "$db\$db.sql"
    # Use the path as it appears inside the container (assuming project is mounted at /workspace)
    $container_db_path = "/workspace/data/database_data/$db/$db.sql"
    if (Test-Path $db_path) {
        Write-Host "Importing $db_path into database $db..."
        docker-compose exec postgres psql -U postgres -d $db -f $container_db_path
        Write-Host "Successfully imported $db_path into database $db"
    } else {
        Write-Host "Warning: $db_path not found, skipping..."
    }
}

Write-Host "PostgreSQL database initialization complete!"
