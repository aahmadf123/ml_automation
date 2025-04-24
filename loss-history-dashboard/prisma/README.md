# Database Setup

This project uses Prisma as an ORM to connect to a PostgreSQL database.

## Requirements

- PostgreSQL database
- Node.js and npm/yarn

## Setup Instructions

1. Create a PostgreSQL database
2. Create a `.env` file in the root of the project with the following content:

```
DATABASE_URL="postgresql://username:password@localhost:5432/database_name?schema=public"
```

Replace `username`, `password`, and `database_name` with your PostgreSQL credentials.

3. Install dependencies:

```bash
npm install
# or
yarn install
```

4. Generate Prisma client:

```bash
npx prisma generate
```

5. Run migrations:

```bash
npx prisma migrate dev
```

## Models

The database currently contains the following models:

### FixTemplate

Stores templates for common fixes:

- `id`: Unique identifier (CUID)
- `name`: Template name
- `description`: Template description (optional)
- `problemType`: Type of problem this template addresses
- `solution`: Solution description
- `code`: Code snippet for the solution (optional)
- `createdAt`: Timestamp when the template was created
- `updatedAt`: Timestamp when the template was last updated 