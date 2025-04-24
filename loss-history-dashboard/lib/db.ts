import AWS from 'aws-sdk';

// Initialize S3 client
const s3 = new AWS.S3({
  region: process.env.AWS_REGION || 'us-east-1',
});

const BUCKET_NAME = 'grange-seniordesign-bucket';
const TEMPLATES_PATH = 'fix-templates/';
const PROPOSALS_PATH = 'fix-proposals/';

// Helper to ensure folder exists in S3
async function ensureFolderExists(folderPath) {
  try {
    // S3 doesn't have real folders, but we can create a zero-byte object with a trailing slash
    await s3.headObject({
      Bucket: BUCKET_NAME,
      Key: folderPath
    }).promise();
  } catch (error) {
    if (error.code === 'NotFound') {
      // Create the folder marker
      await s3.putObject({
        Bucket: BUCKET_NAME,
        Key: folderPath,
        Body: ''
      }).promise();
      console.log(`Created folder ${folderPath} in bucket ${BUCKET_NAME}`);
    } else {
      console.error('Error checking for folder:', error);
      throw error;
    }
  }
}

// S3-based implementation of database operations
export const prisma = {
  // Fix Templates (original implementation)
  fixTemplate: {
    // Find all templates
    async findMany({ orderBy } = {}) {
      await ensureFolderExists(TEMPLATES_PATH);
      
      const response = await s3.listObjects({
        Bucket: BUCKET_NAME,
        Prefix: TEMPLATES_PATH
      }).promise();
      
      const templates = [];
      
      // If there are objects in the folder
      if (response.Contents && response.Contents.length > 0) {
        // Skip the folder marker itself
        const files = response.Contents.filter(item => item.Key !== TEMPLATES_PATH);
        
        // Get content of each file
        for (const file of files) {
          const data = await s3.getObject({
            Bucket: BUCKET_NAME,
            Key: file.Key
          }).promise();
          
          if (data.Body) {
            try {
              const template = JSON.parse(data.Body.toString('utf-8'));
              templates.push(template);
            } catch (e) {
              console.error(`Error parsing template from ${file.Key}:`, e);
            }
          }
        }
      }
      
      // Sort by createdAt if orderBy specifies it
      if (orderBy && orderBy.createdAt === 'desc') {
        templates.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
      }
      
      return templates;
    },
    
    // Find a template by ID
    async findUnique({ where } = {}) {
      if (!where || !where.id) return null;
      
      try {
        const response = await s3.getObject({
          Bucket: BUCKET_NAME,
          Key: `${TEMPLATES_PATH}${where.id}.json`
        }).promise();
        
        if (response.Body) {
          return JSON.parse(response.Body.toString('utf-8'));
        }
        return null;
      } catch (error) {
        if (error.code === 'NoSuchKey') {
          return null;
        }
        throw error;
      }
    },
    
    // Create a new template
    async create({ data } = {}) {
      await ensureFolderExists(TEMPLATES_PATH);
      
      const now = new Date().toISOString();
      const template = {
        id: `template-${Date.now()}`, // Simple ID generation
        ...data,
        createdAt: now,
        updatedAt: now
      };
      
      await s3.putObject({
        Bucket: BUCKET_NAME,
        Key: `${TEMPLATES_PATH}${template.id}.json`,
        Body: JSON.stringify(template),
        ContentType: 'application/json'
      }).promise();
      
      return template;
    },
    
    // Update a template
    async update({ where, data } = {}) {
      if (!where || !where.id) throw new Error('ID is required for update');
      
      // Get the existing template
      const existing = await prisma.fixTemplate.findUnique({ where });
      if (!existing) throw new Error('Template not found');
      
      // Update the template
      const updated = {
        ...existing,
        ...data,
        updatedAt: new Date().toISOString()
      };
      
      await s3.putObject({
        Bucket: BUCKET_NAME,
        Key: `${TEMPLATES_PATH}${where.id}.json`,
        Body: JSON.stringify(updated),
        ContentType: 'application/json'
      }).promise();
      
      return updated;
    },
    
    // Delete a template
    async delete({ where } = {}) {
      if (!where || !where.id) throw new Error('ID is required for deletion');
      
      await s3.deleteObject({
        Bucket: BUCKET_NAME,
        Key: `${TEMPLATES_PATH}${where.id}.json`
      }).promise();
      
      return { id: where.id };
    }
  },
  
  // Fix Proposals (new implementation)
  fixProposal: {
    // Find all proposals
    async findMany({ where, orderBy } = {}) {
      await ensureFolderExists(PROPOSALS_PATH);
      
      const response = await s3.listObjects({
        Bucket: BUCKET_NAME,
        Prefix: PROPOSALS_PATH
      }).promise();
      
      let proposals = [];
      
      // If there are objects in the folder
      if (response.Contents && response.Contents.length > 0) {
        // Skip the folder marker itself
        const files = response.Contents.filter(item => item.Key !== PROPOSALS_PATH);
        
        // Get content of each file
        for (const file of files) {
          const data = await s3.getObject({
            Bucket: BUCKET_NAME,
            Key: file.Key
          }).promise();
          
          if (data.Body) {
            try {
              const proposal = JSON.parse(data.Body.toString('utf-8'));
              proposals.push(proposal);
            } catch (e) {
              console.error(`Error parsing proposal from ${file.Key}:`, e);
            }
          }
        }
      }
      
      // Filter by status if requested
      if (where && where.status) {
        proposals = proposals.filter(p => p.status === where.status);
      }
      
      // Filter by model ID if requested
      if (where && where.modelId) {
        proposals = proposals.filter(p => p.modelId === where.modelId);
      }
      
      // Sort by createdAt if orderBy specifies it
      if (orderBy && orderBy.createdAt === 'desc') {
        proposals.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
      }
      
      return proposals;
    },
    
    // Find a proposal by ID
    async findUnique({ where } = {}) {
      if (!where || !where.id) return null;
      
      try {
        const response = await s3.getObject({
          Bucket: BUCKET_NAME,
          Key: `${PROPOSALS_PATH}${where.id}.json`
        }).promise();
        
        if (response.Body) {
          return JSON.parse(response.Body.toString('utf-8'));
        }
        return null;
      } catch (error) {
        if (error.code === 'NoSuchKey') {
          return null;
        }
        throw error;
      }
    },
    
    // Create a new proposal
    async create({ data } = {}) {
      await ensureFolderExists(PROPOSALS_PATH);
      
      const now = new Date().toISOString();
      const proposal = {
        id: `proposal-${Date.now()}`, // Simple ID generation
        ...data,
        createdAt: now,
        updatedAt: now
      };
      
      await s3.putObject({
        Bucket: BUCKET_NAME,
        Key: `${PROPOSALS_PATH}${proposal.id}.json`,
        Body: JSON.stringify(proposal),
        ContentType: 'application/json'
      }).promise();
      
      return proposal;
    },
    
    // Update a proposal
    async update({ where, data } = {}) {
      if (!where || !where.id) throw new Error('ID is required for update');
      
      // Get the existing proposal
      const existing = await prisma.fixProposal.findUnique({ where });
      if (!existing) throw new Error('Proposal not found');
      
      // Update the proposal
      const updated = {
        ...existing,
        ...data,
        updatedAt: new Date().toISOString()
      };
      
      await s3.putObject({
        Bucket: BUCKET_NAME,
        Key: `${PROPOSALS_PATH}${where.id}.json`,
        Body: JSON.stringify(updated),
        ContentType: 'application/json'
      }).promise();
      
      return updated;
    },
    
    // Delete a proposal
    async delete({ where } = {}) {
      if (!where || !where.id) throw new Error('ID is required for deletion');
      
      await s3.deleteObject({
        Bucket: BUCKET_NAME,
        Key: `${PROPOSALS_PATH}${where.id}.json`
      }).promise();
      
      return { id: where.id };
    }
  }
}; 