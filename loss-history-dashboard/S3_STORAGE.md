# S3 Storage Configuration

This project uses AWS S3 for storing fix templates and proposal data instead of a traditional database.

## Storage Structure

The data is stored in the `grange-seniordesign-bucket` bucket with the following structure:

- `fix-templates/` - Directory for fix template data
  - `{id}.json` - Individual template files
- `fix-proposals/` - Directory for fix proposal data
  - `{id}.json` - Individual proposal files

## Implementation Details

### How it works

- Each database record is stored as a JSON file in S3
- The system creates the appropriate folders if they don't exist
- A serverless approach with no database infrastructure required
- Consistent with the AWS services used in the rest of the application
- Client-side filtering used instead of complex database queries

### Implemented Models

1. **FixTemplate** - Templates for common fixes and solutions
   - Properties: id, name, description, problemType, solution, code, createdAt, updatedAt

2. **FixProposal** - Proposed fixes that need approval
   - Properties: id, problem, solution, status, modelId, createdAt, updatedAt

### Limitations compared to a database

- No complex queries or joins (everything is file-based)
- Limited transaction support
- Limited indexing capabilities
- Possible higher latency for large datasets

## AWS Configuration Requirements

Ensure your AWS credentials have the following permissions for the bucket:

- `s3:ListBucket`
- `s3:GetObject`
- `s3:PutObject`
- `s3:DeleteObject`

## Environment Setup

Make sure your application has the following environment variables:

```env
AWS_REGION=us-east-1 # or your preferred region
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
```

You can add these to your `.env.local` file for local development.

## Extending the Storage

If you need to add additional data models:

1. Create a new folder in the S3 bucket for the model
2. Extend the `db.ts` file with the new model implementation
3. Follow the same pattern as the existing implementations

For example:

```javascript
// In db.ts
export const prisma = {
  fixTemplate: { /* existing implementation */ },
  fixProposal: { /* existing implementation */ },
  newModel: {
    async findMany() { /* implementation */ },
    async findUnique() { /* implementation */ },
    async create() { /* implementation */ },
    async update() { /* implementation */ },
    async delete() { /* implementation */ }
  }
}
``` 